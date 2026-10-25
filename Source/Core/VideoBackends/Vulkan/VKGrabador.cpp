// Copyright 2026 STV
// SPDX-License-Identifier: GPL-2.0-or-later

// STV_GC_GRABADOR_v1: ver VKGrabador.h.

#include "VideoBackends/Vulkan/VKGrabador.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <pthread.h>

#include "VideoBackends/Vulkan/VulkanLoader.h"
#include "VideoCommon/Fifo.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#define STV_LOG(...) __android_log_print(ANDROID_LOG_INFO, "STVGC", __VA_ARGS__)
#else
#define STV_LOG(...) ((void)0)
#endif

namespace Vulkan::Grabador
{
bool g_limpieza = false;

namespace
{
// ---- anillo de un productor (hilo de video) y un consumidor (grabador) ----
constexpr size_t CAP = 4u << 20;  // 4 MB: un cuadro de Twilight Princess graba bastante menos
constexpr size_t CAB = 16;        // cabecera: tamano total de la entrada + funcion

struct Cabecera
{
  u64 tam;  // bytes de la entrada entera, cabecera incluida, multiplo de 16
  void (*fn)(u8*);  // nullptr = relleno hasta el final del anillo
};

u8* s_buf = nullptr;
std::atomic<u64> s_w{0};  // publicado por el productor
std::atomic<u64> s_r{0};  // consumido por el grabador
u64 s_wl = 0;             // posicion local del productor (reservado sin publicar)
u64 s_tam_pend = 0;

std::atomic<bool> s_dormido{false};   // el grabador duerme esperando trabajo
std::atomic<u32> s_aviso{0};          // futex: hay trabajo
std::atomic<bool> s_espera{false};    // el productor duerme esperando que el grabador avance
std::atomic<u32> s_hecho{0};          // futex: el grabador avanzo
std::atomic<bool> s_salir{false};
std::atomic<bool> s_activo{false};
bool s_enganchado = false;
std::thread s_hilo;
u32 s_releer = 0;

constexpr size_t Al16(size_t n)
{
  return (n + 15) & ~size_t(15);
}

bool EsHiloVideo()
{
  const pthread_t v = Fifo::g_stv_hilo_gpu.load(std::memory_order_relaxed);
  return v != 0 && pthread_equal(v, pthread_self());
}

inline void Ceder()
{
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

// El productor espera a que el grabador consuma hasta `meta` (vaciar o hacer lugar).
void EsperarConsumo(u64 meta)
{
  for (int i = 0; i < 2000; i++)
  {
    if (s_r.load(std::memory_order_acquire) >= meta)
      return;
    Ceder();
  }
  for (;;)
  {
    const u32 h = s_hecho.load(std::memory_order_seq_cst);
    s_espera.store(true, std::memory_order_seq_cst);
    if (s_r.load(std::memory_order_seq_cst) >= meta)
      break;
    s_hecho.wait(h, std::memory_order_seq_cst);
  }
  s_espera.store(false, std::memory_order_relaxed);
}

void Hilo()
{
  pthread_setname_np(pthread_self(), "STV grabador");
  u64 r = s_r.load(std::memory_order_relaxed);
  for (;;)
  {
    u64 w = s_w.load(std::memory_order_acquire);
    if (r == w)
    {
      // Sin trabajo: unas vueltas cortas (el hilo de video publica en rafagas) y despues dormir.
      bool hay = false;
      for (int i = 0; i < 1500 && !hay; i++)
      {
        Ceder();
        hay = s_w.load(std::memory_order_acquire) != r;
      }
      if (!hay)
      {
        if (s_salir.load(std::memory_order_acquire))
          break;
        const u32 a = s_aviso.load(std::memory_order_seq_cst);
        s_dormido.store(true, std::memory_order_seq_cst);
        if (s_w.load(std::memory_order_seq_cst) == r && !s_salir.load(std::memory_order_seq_cst))
          s_aviso.wait(a, std::memory_order_seq_cst);
        s_dormido.store(false, std::memory_order_relaxed);
      }
      continue;
    }
    while (r != w)
    {
      const size_t pos = r % CAP;
      const Cabecera* c = reinterpret_cast<const Cabecera*>(s_buf + pos);
      if (c->fn)
        c->fn(s_buf + pos + CAB);
      r += c->tam;
      s_r.store(r, std::memory_order_release);
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (s_espera.load(std::memory_order_seq_cst))
    {
      s_hecho.fetch_add(1, std::memory_order_seq_cst);
      s_hecho.notify_all();
    }
  }
}

// ---- los punteros reales ----
#define STV_REALES(X)                                                                              \
  X(vkCmdBindPipeline)                                                                             \
  X(vkCmdSetViewport)                                                                              \
  X(vkCmdSetScissor)                                                                               \
  X(vkCmdBindDescriptorSets)                                                                       \
  X(vkCmdBindIndexBuffer)                                                                          \
  X(vkCmdBindVertexBuffers)                                                                        \
  X(vkCmdDraw)                                                                                     \
  X(vkCmdDrawIndexed)                                                                              \
  X(vkCmdDispatch)                                                                                 \
  X(vkCmdCopyBuffer)                                                                               \
  X(vkCmdCopyImage)                                                                                \
  X(vkCmdBlitImage)                                                                                \
  X(vkCmdCopyBufferToImage)                                                                        \
  X(vkCmdCopyImageToBuffer)                                                                        \
  X(vkCmdUpdateBuffer)                                                                             \
  X(vkCmdClearAttachments)                                                                         \
  X(vkCmdResolveImage)                                                                             \
  X(vkCmdPipelineBarrier)                                                                          \
  X(vkCmdBeginQuery)                                                                               \
  X(vkCmdEndQuery)                                                                                 \
  X(vkCmdResetQueryPool)                                                                           \
  X(vkCmdBeginRenderPass)                                                                          \
  X(vkCmdEndRenderPass)                                                                            \
  X(vkBeginCommandBuffer)                                                                          \
  X(vkEndCommandBuffer)                                                                            \
  X(vkResetCommandPool)                                                                            \
  X(vkUpdateDescriptorSets)                                                                        \
  /* sin diferir: vaciar y llamar directo */                                                       \
  X(vkQueueSubmit)                                                                                 \
  X(vkQueueWaitIdle)                                                                               \
  X(vkDeviceWaitIdle)                                                                              \
  X(vkAcquireNextImageKHR)                                                                         \
  X(vkDestroySwapchainKHR)                                                                         \
  X(vkDestroyQueryPool)                                                                            \
  X(vkDestroyBuffer)                                                                               \
  X(vkDestroyBufferView)                                                                           \
  X(vkDestroyImage)                                                                                \
  X(vkDestroyImageView)                                                                            \
  X(vkDestroyPipeline)                                                                             \
  X(vkDestroySampler)                                                                              \
  X(vkDestroyDescriptorPool)                                                                       \
  X(vkDestroyFramebuffer)                                                                          \
  X(vkDestroyRenderPass)                                                                           \
  /* comandos que Dolphin 2609 no usa: vaciar, avisar una vez y llamar directo */                  \
  X(vkCmdSetLineWidth)                                                                             \
  X(vkCmdSetDepthBias)                                                                             \
  X(vkCmdSetBlendConstants)                                                                        \
  X(vkCmdSetDepthBounds)                                                                           \
  X(vkCmdSetStencilCompareMask)                                                                    \
  X(vkCmdSetStencilWriteMask)                                                                      \
  X(vkCmdSetStencilReference)                                                                      \
  X(vkCmdDrawIndirect)                                                                             \
  X(vkCmdDrawIndexedIndirect)                                                                      \
  X(vkCmdDispatchIndirect)                                                                         \
  X(vkCmdFillBuffer)                                                                               \
  X(vkCmdClearColorImage)                                                                          \
  X(vkCmdClearDepthStencilImage)                                                                   \
  X(vkCmdSetEvent)                                                                                 \
  X(vkCmdResetEvent)                                                                               \
  X(vkCmdWaitEvents)                                                                               \
  X(vkCmdWriteTimestamp)                                                                           \
  X(vkCmdCopyQueryPoolResults)                                                                     \
  X(vkCmdPushConstants)                                                                            \
  X(vkCmdNextSubpass)                                                                              \
  X(vkCmdExecuteCommands)

#define STV_DECL(n) PFN_##n r_##n = nullptr;
STV_REALES(STV_DECL)
#undef STV_DECL

// Copia de arreglos dentro de la carga de una entrada.
struct Carga
{
  u8* p;
  template <typename T>
  T* Copia(const T* src, u32 n)
  {
    if (!src || !n)
      return nullptr;
    T* d = reinterpret_cast<T*>(p);
    std::memcpy(d, src, sizeof(T) * n);
    p += Al16(sizeof(T) * n);
    return d;
  }
};
template <typename T>
constexpr size_t B(u32 n)
{
  return Al16(sizeof(T) * n);
}

// Arma una entrada con la estructura A (argumentos) + arreglos de `extra` bytes; f la ejecuta en el grabador.
template <typename A>
A* Nueva(void (*f)(u8*), size_t extra, Carga* c)
{
  u8* p = Reservar(f, Al16(sizeof(A)) + extra);
  c->p = p + Al16(sizeof(A));
  return reinterpret_cast<A*>(p);
}

// Cuando la llamada no se difiere: si la hace otro hilo con algo pendiente, primero se vacia.
inline void Directo()
{
  if (s_r.load(std::memory_order_acquire) != s_w.load(std::memory_order_acquire) && !EsHiloVideo())
    Vaciar();
}

// ================== comandos diferidos ==================
VKAPI_ATTR void VKAPI_CALL t_vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pl)
{
  if (!Difiere())
    return Directo(), r_vkCmdBindPipeline(cb, bp, pl);
  struct A { VkCommandBuffer cb; VkPipelineBindPoint bp; VkPipeline pl; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdBindPipeline(a->cb, a->bp, a->pl); }, 0, &c);
  *a = {cb, bp, pl};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v)
{
  if (!Difiere())
    return Directo(), r_vkCmdSetViewport(cb, first, n, v);
  struct A { VkCommandBuffer cb; uint32_t first, n; const VkViewport* v; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdSetViewport(a->cb, a->first, a->n, a->v); },
                  B<VkViewport>(n), &c);
  *a = {cb, first, n, c.Copia(v, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* s)
{
  if (!Difiere())
    return Directo(), r_vkCmdSetScissor(cb, first, n, s);
  struct A { VkCommandBuffer cb; uint32_t first, n; const VkRect2D* s; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdSetScissor(a->cb, a->first, a->n, a->s); },
                  B<VkRect2D>(n), &c);
  *a = {cb, first, n, c.Copia(s, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                     VkPipelineLayout lay, uint32_t first, uint32_t ns,
                                                     const VkDescriptorSet* sets, uint32_t nd, const uint32_t* dyn)
{
  if (!Difiere())
    return Directo(), r_vkCmdBindDescriptorSets(cb, bp, lay, first, ns, sets, nd, dyn);
  struct A { VkCommandBuffer cb; VkPipelineBindPoint bp; VkPipelineLayout lay; uint32_t first, ns;
             const VkDescriptorSet* sets; uint32_t nd; const uint32_t* dyn; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdBindDescriptorSets(a->cb, a->bp, a->lay, a->first, a->ns, a->sets, a->nd, a->dyn);
  }, B<VkDescriptorSet>(ns) + B<uint32_t>(nd), &c);
  a->cb = cb; a->bp = bp; a->lay = lay; a->first = first; a->ns = ns; a->nd = nd;
  a->sets = c.Copia(sets, ns);
  a->dyn = c.Copia(dyn, nd);
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkIndexType t)
{
  if (!Difiere())
    return Directo(), r_vkCmdBindIndexBuffer(cb, b, off, t);
  struct A { VkCommandBuffer cb; VkBuffer b; VkDeviceSize off; VkIndexType t; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdBindIndexBuffer(a->cb, a->b, a->off, a->t); }, 0, &c);
  *a = {cb, b, off, t};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                    const VkBuffer* bufs, const VkDeviceSize* offs)
{
  if (!Difiere())
    return Directo(), r_vkCmdBindVertexBuffers(cb, first, n, bufs, offs);
  struct A { VkCommandBuffer cb; uint32_t first, n; const VkBuffer* bufs; const VkDeviceSize* offs; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdBindVertexBuffers(a->cb, a->first, a->n, a->bufs, a->offs);
  }, B<VkBuffer>(n) + B<VkDeviceSize>(n), &c);
  a->cb = cb; a->first = first; a->n = n;
  a->bufs = c.Copia(bufs, n);
  a->offs = c.Copia(offs, n);
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdDraw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi)
{
  if (!Difiere())
    return Directo(), r_vkCmdDraw(cb, vc, ic, fv, fi);
  struct A { VkCommandBuffer cb; uint32_t vc, ic, fv, fi; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdDraw(a->cb, a->vc, a->ic, a->fv, a->fi); }, 0, &c);
  *a = {cb, vc, ic, fv, fi};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo,
                                              uint32_t finst)
{
  if (!Difiere())
    return Directo(), r_vkCmdDrawIndexed(cb, ic, inst, fi, vo, finst);
  struct A { VkCommandBuffer cb; uint32_t ic, inst, fi; int32_t vo; uint32_t finst; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdDrawIndexed(a->cb, a->ic, a->inst, a->fi, a->vo, a->finst);
  }, 0, &c);
  *a = {cb, ic, inst, fi, vo, finst};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z)
{
  if (!Difiere())
    return Directo(), r_vkCmdDispatch(cb, x, y, z);
  struct A { VkCommandBuffer cb; uint32_t x, y, z; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdDispatch(a->cb, a->x, a->y, a->z); }, 0, &c);
  *a = {cb, x, y, z};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n,
                                             const VkBufferCopy* r)
{
  if (!Difiere())
    return Directo(), r_vkCmdCopyBuffer(cb, s, d, n, r);
  struct A { VkCommandBuffer cb; VkBuffer s, d; uint32_t n; const VkBufferCopy* r; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdCopyBuffer(a->cb, a->s, a->d, a->n, a->r); },
                  B<VkBufferCopy>(n), &c);
  *a = {cb, s, d, n, c.Copia(r, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d,
                                            VkImageLayout dl, uint32_t n, const VkImageCopy* r)
{
  if (!Difiere())
    return Directo(), r_vkCmdCopyImage(cb, s, sl, d, dl, n, r);
  struct A { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n;
             const VkImageCopy* r; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdCopyImage(a->cb, a->s, a->sl, a->d, a->dl, a->n, a->r);
  }, B<VkImageCopy>(n), &c);
  *a = {cb, s, sl, d, dl, n, c.Copia(r, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d,
                                            VkImageLayout dl, uint32_t n, const VkImageBlit* r, VkFilter f)
{
  if (!Difiere())
    return Directo(), r_vkCmdBlitImage(cb, s, sl, d, dl, n, r, f);
  struct A { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n;
             const VkImageBlit* r; VkFilter f; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdBlitImage(a->cb, a->s, a->sl, a->d, a->dl, a->n, a->r, a->f);
  }, B<VkImageBlit>(n), &c);
  *a = {cb, s, sl, d, dl, n, c.Copia(r, n), f};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout dl,
                                                    uint32_t n, const VkBufferImageCopy* r)
{
  if (!Difiere())
    return Directo(), r_vkCmdCopyBufferToImage(cb, s, d, dl, n, r);
  struct A { VkCommandBuffer cb; VkBuffer s; VkImage d; VkImageLayout dl; uint32_t n; const VkBufferImageCopy* r; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdCopyBufferToImage(a->cb, a->s, a->d, a->dl, a->n, a->r);
  }, B<VkBufferImageCopy>(n), &c);
  *a = {cb, s, d, dl, n, c.Copia(r, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkBuffer d,
                                                    uint32_t n, const VkBufferImageCopy* r)
{
  if (!Difiere())
    return Directo(), r_vkCmdCopyImageToBuffer(cb, s, sl, d, n, r);
  struct A { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkBuffer d; uint32_t n; const VkBufferImageCopy* r; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdCopyImageToBuffer(a->cb, a->s, a->sl, a->d, a->n, a->r);
  }, B<VkBufferImageCopy>(n), &c);
  *a = {cb, s, sl, d, n, c.Copia(r, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer d, VkDeviceSize off, VkDeviceSize n,
                                               const void* data)
{
  if (!Difiere())
    return Directo(), r_vkCmdUpdateBuffer(cb, d, off, n, data);
  struct A { VkCommandBuffer cb; VkBuffer d; VkDeviceSize off, n; const void* data; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdUpdateBuffer(a->cb, a->d, a->off, a->n, a->data); },
                  B<u8>(static_cast<u32>(n)), &c);
  *a = {cb, d, off, n, c.Copia(static_cast<const u8*>(data), static_cast<u32>(n))};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment* at,
                                                   uint32_t nr, const VkClearRect* rc)
{
  if (!Difiere())
    return Directo(), r_vkCmdClearAttachments(cb, na, at, nr, rc);
  struct A { VkCommandBuffer cb; uint32_t na; const VkClearAttachment* at; uint32_t nr; const VkClearRect* rc; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdClearAttachments(a->cb, a->na, a->at, a->nr, a->rc);
  }, B<VkClearAttachment>(na) + B<VkClearRect>(nr), &c);
  a->cb = cb; a->na = na; a->nr = nr;
  a->at = c.Copia(at, na);
  a->rc = c.Copia(rc, nr);
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d,
                                               VkImageLayout dl, uint32_t n, const VkImageResolve* r)
{
  if (!Difiere())
    return Directo(), r_vkCmdResolveImage(cb, s, sl, d, dl, n, r);
  struct A { VkCommandBuffer cb; VkImage s; VkImageLayout sl; VkImage d; VkImageLayout dl; uint32_t n;
             const VkImageResolve* r; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdResolveImage(a->cb, a->s, a->sl, a->d, a->dl, a->n, a->r);
  }, B<VkImageResolve>(n), &c);
  *a = {cb, s, sl, d, dl, n, c.Copia(r, n)};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss,
                                                  VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t nm,
                                                  const VkMemoryBarrier* mb, uint32_t nb,
                                                  const VkBufferMemoryBarrier* bb, uint32_t ni,
                                                  const VkImageMemoryBarrier* ib)
{
  bool pnext = false;
  for (uint32_t i = 0; i < nm; i++) pnext |= mb[i].pNext != nullptr;
  for (uint32_t i = 0; i < nb; i++) pnext |= bb[i].pNext != nullptr;
  for (uint32_t i = 0; i < ni; i++) pnext |= ib[i].pNext != nullptr;
  if (!Difiere() || pnext)
  {
    if (pnext && Difiere())
      Vaciar();
    return Directo(), r_vkCmdPipelineBarrier(cb, ss, ds, df, nm, mb, nb, bb, ni, ib);
  }
  struct A { VkCommandBuffer cb; VkPipelineStageFlags ss, ds; VkDependencyFlags df; uint32_t nm;
             const VkMemoryBarrier* mb; uint32_t nb; const VkBufferMemoryBarrier* bb; uint32_t ni;
             const VkImageMemoryBarrier* ib; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkCmdPipelineBarrier(a->cb, a->ss, a->ds, a->df, a->nm, a->mb, a->nb, a->bb, a->ni, a->ib);
  }, B<VkMemoryBarrier>(nm) + B<VkBufferMemoryBarrier>(nb) + B<VkImageMemoryBarrier>(ni), &c);
  a->cb = cb; a->ss = ss; a->ds = ds; a->df = df; a->nm = nm; a->nb = nb; a->ni = ni;
  a->mb = c.Copia(mb, nm);
  a->bb = c.Copia(bb, nb);
  a->ib = c.Copia(ib, ni);
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q, VkQueryControlFlags f)
{
  if (!Difiere())
    return Directo(), r_vkCmdBeginQuery(cb, qp, q, f);
  struct A { VkCommandBuffer cb; VkQueryPool qp; uint32_t q; VkQueryControlFlags f; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdBeginQuery(a->cb, a->qp, a->q, a->f); }, 0, &c);
  *a = {cb, qp, q, f};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool qp, uint32_t q)
{
  if (!Difiere())
    return Directo(), r_vkCmdEndQuery(cb, qp, q);
  struct A { VkCommandBuffer cb; VkQueryPool qp; uint32_t q; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdEndQuery(a->cb, a->qp, a->q); }, 0, &c);
  *a = {cb, qp, q};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool qp, uint32_t first, uint32_t n)
{
  if (!Difiere())
    return Directo(), r_vkCmdResetQueryPool(cb, qp, first, n);
  struct A { VkCommandBuffer cb; VkQueryPool qp; uint32_t first, n; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdResetQueryPool(a->cb, a->qp, a->first, a->n); }, 0, &c);
  *a = {cb, qp, first, n};
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* bi,
                                                  VkSubpassContents sc)
{
  if (!Difiere() || bi->pNext)
  {
    if (bi->pNext && Difiere())
      Vaciar();
    return Directo(), r_vkCmdBeginRenderPass(cb, bi, sc);
  }
  struct A { VkCommandBuffer cb; VkRenderPassBeginInfo bi; VkSubpassContents sc; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { A* a = reinterpret_cast<A*>(p); r_vkCmdBeginRenderPass(a->cb, &a->bi, a->sc); },
                  B<VkClearValue>(bi->clearValueCount), &c);
  a->cb = cb; a->sc = sc; a->bi = *bi;
  a->bi.pClearValues = c.Copia(bi->pClearValues, bi->clearValueCount);
  Publicar();
}

VKAPI_ATTR void VKAPI_CALL t_vkCmdEndRenderPass(VkCommandBuffer cb)
{
  if (!Difiere())
    return Directo(), r_vkCmdEndRenderPass(cb);
  struct A { VkCommandBuffer cb; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { r_vkCmdEndRenderPass(reinterpret_cast<A*>(p)->cb); }, 0, &c);
  a->cb = cb;
  Publicar();
}

void Fallo(const char* que, VkResult r)
{
  if (r != VK_SUCCESS)
    STV_LOG("grabador: %s fallo (%d)", que, static_cast<int>(r));
}

VKAPI_ATTR VkResult VKAPI_CALL t_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* bi)
{
  if (!Difiere() || bi->pNext || bi->pInheritanceInfo)
  {
    if ((bi->pNext || bi->pInheritanceInfo) && Difiere())
      Vaciar();
    Directo();
    return r_vkBeginCommandBuffer(cb, bi);
  }
  struct A { VkCommandBuffer cb; VkCommandBufferBeginInfo bi; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    Fallo("vkBeginCommandBuffer", r_vkBeginCommandBuffer(a->cb, &a->bi));
  }, 0, &c);
  a->cb = cb; a->bi = *bi;
  Publicar();
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL t_vkEndCommandBuffer(VkCommandBuffer cb)
{
  if (!Difiere())
  {
    Directo();
    return r_vkEndCommandBuffer(cb);
  }
  struct A { VkCommandBuffer cb; };
  Carga c;
  A* a = Nueva<A>([](u8* p) { Fallo("vkEndCommandBuffer", r_vkEndCommandBuffer(reinterpret_cast<A*>(p)->cb)); }, 0, &c);
  a->cb = cb;
  Publicar();
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL t_vkResetCommandPool(VkDevice dev, VkCommandPool pool, VkCommandPoolResetFlags f)
{
  if (!Difiere())
  {
    Directo();
    return r_vkResetCommandPool(dev, pool, f);
  }
  struct A { VkDevice dev; VkCommandPool pool; VkCommandPoolResetFlags f; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    Fallo("vkResetCommandPool", r_vkResetCommandPool(a->dev, a->pool, a->f));
  }, 0, &c);
  *a = {dev, pool, f};
  Publicar();
  return VK_SUCCESS;
}

bool EsImagen(VkDescriptorType t)
{
  return t == VK_DESCRIPTOR_TYPE_SAMPLER || t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
         t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE || t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
         t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}
bool EsBufer(VkDescriptorType t)
{
  return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
         t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}
bool EsTexel(VkDescriptorType t)
{
  return t == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || t == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
}

VKAPI_ATTR void VKAPI_CALL t_vkUpdateDescriptorSets(VkDevice dev, uint32_t nw, const VkWriteDescriptorSet* w,
                                                    uint32_t nc, const VkCopyDescriptorSet* cp)
{
  bool raro = false;
  size_t extra = B<VkWriteDescriptorSet>(nw) + B<VkCopyDescriptorSet>(nc);
  for (uint32_t i = 0; i < nw; i++)
  {
    const VkDescriptorType t = w[i].descriptorType;
    raro |= w[i].pNext != nullptr || !(EsImagen(t) || EsBufer(t) || EsTexel(t));
    if (EsImagen(t)) extra += B<VkDescriptorImageInfo>(w[i].descriptorCount);
    if (EsBufer(t)) extra += B<VkDescriptorBufferInfo>(w[i].descriptorCount);
    if (EsTexel(t)) extra += B<VkBufferView>(w[i].descriptorCount);
  }
  for (uint32_t i = 0; i < nc; i++)
    raro |= cp[i].pNext != nullptr;
  if (!Difiere() || raro)
  {
    if (raro && Difiere())
      Vaciar();
    return Directo(), r_vkUpdateDescriptorSets(dev, nw, w, nc, cp);
  }
  struct A { VkDevice dev; uint32_t nw; VkWriteDescriptorSet* w; uint32_t nc; const VkCopyDescriptorSet* cp; };
  Carga c;
  A* a = Nueva<A>([](u8* p) {
    A* a = reinterpret_cast<A*>(p);
    r_vkUpdateDescriptorSets(a->dev, a->nw, a->w, a->nc, a->cp);
  }, extra, &c);
  a->dev = dev; a->nw = nw; a->nc = nc;
  a->w = c.Copia(w, nw);
  a->cp = c.Copia(cp, nc);
  for (uint32_t i = 0; i < nw; i++)
  {
    VkWriteDescriptorSet& x = a->w[i];
    const VkDescriptorType t = x.descriptorType;
    x.pImageInfo = EsImagen(t) ? c.Copia(w[i].pImageInfo, x.descriptorCount) : nullptr;
    x.pBufferInfo = EsBufer(t) ? c.Copia(w[i].pBufferInfo, x.descriptorCount) : nullptr;
    x.pTexelBufferView = EsTexel(t) ? c.Copia(w[i].pTexelBufferView, x.descriptorCount) : nullptr;
  }
  Publicar();
}

// ================== sin diferir: vaciar en el hilo de video y llamar directo ==================
inline void Antes()
{
  if (EsHiloVideo())
    Vaciar();
}
inline void AntesDestruir()
{
  if (EsHiloVideo() && !g_limpieza)
    Vaciar();
}

VKAPI_ATTR VkResult VKAPI_CALL t_vkQueueSubmit(VkQueue q, uint32_t n, const VkSubmitInfo* s, VkFence f)
{
  Antes();
  return r_vkQueueSubmit(q, n, s, f);
}
VKAPI_ATTR VkResult VKAPI_CALL t_vkQueueWaitIdle(VkQueue q)
{
  Antes();
  return r_vkQueueWaitIdle(q);
}
VKAPI_ATTR VkResult VKAPI_CALL t_vkDeviceWaitIdle(VkDevice d)
{
  Vaciar();
  return r_vkDeviceWaitIdle(d);
}
VKAPI_ATTR VkResult VKAPI_CALL t_vkAcquireNextImageKHR(VkDevice d, VkSwapchainKHR s, uint64_t t, VkSemaphore se,
                                                      VkFence f, uint32_t* i)
{
  Antes();
  return r_vkAcquireNextImageKHR(d, s, t, se, f, i);
}
#define STV_DESTRUYE(nombre, Tipo)                                                                 \
  VKAPI_ATTR void VKAPI_CALL t_##nombre(VkDevice d, Tipo o, const VkAllocationCallbacks* a)        \
  {                                                                                                \
    AntesDestruir();                                                                               \
    r_##nombre(d, o, a);                                                                           \
  }
STV_DESTRUYE(vkDestroySwapchainKHR, VkSwapchainKHR)
STV_DESTRUYE(vkDestroyQueryPool, VkQueryPool)
STV_DESTRUYE(vkDestroyBuffer, VkBuffer)
STV_DESTRUYE(vkDestroyBufferView, VkBufferView)
STV_DESTRUYE(vkDestroyImage, VkImage)
STV_DESTRUYE(vkDestroyImageView, VkImageView)
STV_DESTRUYE(vkDestroyPipeline, VkPipeline)
STV_DESTRUYE(vkDestroySampler, VkSampler)
STV_DESTRUYE(vkDestroyDescriptorPool, VkDescriptorPool)
STV_DESTRUYE(vkDestroyFramebuffer, VkFramebuffer)
STV_DESTRUYE(vkDestroyRenderPass, VkRenderPass)
#undef STV_DESTRUYE

// ================== comandos que Dolphin no usa: vaciar, avisar una vez y llamar directo ==================
void NoPrevisto(const char* n)
{
  static std::atomic<int> avisos{0};
  if (Difiere())
  {
    Vaciar();
    if (avisos.fetch_add(1) < 8)
      STV_LOG("grabador: %s sin diferir (vaciado y directo)", n);
  }
  else
  {
    Directo();
  }
}
#define STV_NP(nombre, firma, args)                                                                \
  VKAPI_ATTR void VKAPI_CALL t_##nombre firma                                                      \
  {                                                                                                \
    NoPrevisto(#nombre);                                                                           \
    r_##nombre args;                                                                               \
  }
STV_NP(vkCmdSetLineWidth, (VkCommandBuffer cb, float w), (cb, w))
STV_NP(vkCmdSetDepthBias, (VkCommandBuffer cb, float a, float b, float c), (cb, a, b, c))
STV_NP(vkCmdSetBlendConstants, (VkCommandBuffer cb, const float k[4]), (cb, k))
STV_NP(vkCmdSetDepthBounds, (VkCommandBuffer cb, float a, float b), (cb, a, b))
STV_NP(vkCmdSetStencilCompareMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t m), (cb, f, m))
STV_NP(vkCmdSetStencilWriteMask, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t m), (cb, f, m))
STV_NP(vkCmdSetStencilReference, (VkCommandBuffer cb, VkStencilFaceFlags f, uint32_t m), (cb, f, m))
STV_NP(vkCmdDrawIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (cb, b, o, n, s))
STV_NP(vkCmdDrawIndexedIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s),
       (cb, b, o, n, s))
STV_NP(vkCmdDispatchIndirect, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o), (cb, b, o))
STV_NP(vkCmdFillBuffer, (VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize s, uint32_t d), (cb, b, o, s, d))
STV_NP(vkCmdClearColorImage,
       (VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue* v, uint32_t n,
        const VkImageSubresourceRange* r),
       (cb, i, l, v, n, r))
STV_NP(vkCmdClearDepthStencilImage,
       (VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* v, uint32_t n,
        const VkImageSubresourceRange* r),
       (cb, i, l, v, n, r))
STV_NP(vkCmdSetEvent, (VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags m), (cb, e, m))
STV_NP(vkCmdResetEvent, (VkCommandBuffer cb, VkEvent e, VkPipelineStageFlags m), (cb, e, m))
STV_NP(vkCmdWaitEvents,
       (VkCommandBuffer cb, uint32_t ne, const VkEvent* e, VkPipelineStageFlags s, VkPipelineStageFlags d,
        uint32_t nm, const VkMemoryBarrier* mb, uint32_t nb, const VkBufferMemoryBarrier* bb, uint32_t ni,
        const VkImageMemoryBarrier* ib),
       (cb, ne, e, s, d, nm, mb, nb, bb, ni, ib))
STV_NP(vkCmdWriteTimestamp, (VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool p, uint32_t q), (cb, s, p, q))
STV_NP(vkCmdCopyQueryPoolResults,
       (VkCommandBuffer cb, VkQueryPool p, uint32_t f, uint32_t n, VkBuffer b, VkDeviceSize o, VkDeviceSize s,
        VkQueryResultFlags fl),
       (cb, p, f, n, b, o, s, fl))
STV_NP(vkCmdPushConstants,
       (VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags s, uint32_t o, uint32_t n, const void* v),
       (cb, l, s, o, n, v))
STV_NP(vkCmdNextSubpass, (VkCommandBuffer cb, VkSubpassContents c), (cb, c))
STV_NP(vkCmdExecuteCommands, (VkCommandBuffer cb, uint32_t n, const VkCommandBuffer* b), (cb, n, b))
#undef STV_NP
}  // namespace

bool Difiere()
{
  return s_activo.load(std::memory_order_relaxed) && EsHiloVideo();
}

u8* Reservar(void (*fn)(u8*), size_t bytes)
{
  const size_t tot = Al16(CAB + bytes);
  size_t pos = s_wl % CAP;
  if (CAP - pos < tot)
  {
    // Relleno hasta el final del anillo y la entrada al principio.
    const size_t relleno = CAP - pos;
    if (s_wl + relleno + tot > CAP)
      EsperarConsumo(s_wl + relleno + tot - CAP);
    Cabecera* r = reinterpret_cast<Cabecera*>(s_buf + pos);
    r->tam = relleno;
    r->fn = nullptr;
    s_wl += relleno;
    pos = 0;
  }
  else if (s_wl + tot > CAP)
  {
    EsperarConsumo(s_wl + tot - CAP);  // libre = CAP - (s_wl - s_r) tiene que alcanzar para tot
  }
  Cabecera* c = reinterpret_cast<Cabecera*>(s_buf + pos);
  c->tam = tot;
  c->fn = fn;
  s_tam_pend = tot;
  return s_buf + pos + CAB;
}

void Publicar()
{
  s_wl += s_tam_pend;
  s_w.store(s_wl, std::memory_order_seq_cst);
  if (s_dormido.load(std::memory_order_seq_cst))
  {
    s_aviso.fetch_add(1, std::memory_order_seq_cst);
    s_aviso.notify_one();
  }
}

void Vaciar()
{
  if (!s_buf)
    return;
  const u64 meta = s_w.load(std::memory_order_acquire);
  if (s_r.load(std::memory_order_acquire) >= meta)
    return;
  EsperarConsumo(meta);
}

void Releer()
{
  if (!s_enganchado || (s_releer++ % 30) != 0)
    return;
  bool activo = true;
#ifdef __ANDROID__
  char v[PROP_VALUE_MAX] = {};
  if (__system_property_get("debug.stv.gc.grabador", v) > 0 && v[0] == '0')
    activo = false;
#endif
  if (activo == s_activo.load(std::memory_order_relaxed))
    return;
  if (!activo)
    Vaciar();
  s_activo.store(activo, std::memory_order_relaxed);
  STV_LOG("grabador %s", activo ? "activo" : "apagado");
}

void Iniciar()
{
  if (s_enganchado)
    Terminar();
  if (!s_buf)
  {
    void* m = nullptr;
    if (posix_memalign(&m, 64, CAP) != 0)
    {
      STV_LOG("grabador: sin memoria para el anillo, queda apagado");
      return;
    }
    s_buf = static_cast<u8*>(m);
  }
#define STV_ENGANCHA(n)                                                                            \
  r_##n = n;                                                                                       \
  if (n)                                                                                           \
    n = t_##n;
  STV_REALES(STV_ENGANCHA)
#undef STV_ENGANCHA
  s_w.store(0);
  s_r.store(0);
  s_wl = 0;
  s_salir.store(false);
  s_hilo = std::thread(Hilo);
  s_enganchado = true;
  s_releer = 0;
  s_activo.store(true);
  STV_LOG("grabador iniciado (anillo %zu KB)", CAP / 1024);
}

void Terminar()
{
  if (!s_enganchado)
    return;
  s_activo.store(false);
  Vaciar();
  s_salir.store(true, std::memory_order_seq_cst);
  s_aviso.fetch_add(1, std::memory_order_seq_cst);
  s_aviso.notify_one();
  if (s_hilo.joinable())
    s_hilo.join();
#define STV_DESENGANCHA(n)                                                                         \
  if (r_##n)                                                                                       \
    n = r_##n;
  STV_REALES(STV_DESENGANCHA)
#undef STV_DESENGANCHA
  s_enganchado = false;
  STV_LOG("grabador terminado");
}
}  // namespace Vulkan::Grabador
