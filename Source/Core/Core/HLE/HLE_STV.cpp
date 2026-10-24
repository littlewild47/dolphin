// STV_GC_SKINHLE_v1: reemplazo nativo del skinning de vertices por CPU de The Simpsons: Hit & Run (GHQE7D).
//
// La funcion en 0x8027d508 (escrita a mano en asm por el juego) deforma cada vertice con hasta 3 matrices de huesos y
// sus pesos y manda x, y, z al FIFO de la GPU (0xCC008000). Con el auto acelerando es la funcion mas caliente del juego
// (3 % del hilo de CPU de Dolphin). Aca se ejecuta lo MISMO en C++: mismo orden de operaciones, misma aritmetica de
// simple precision (ps_madds = fmaf, un redondeo; ps_muls/ps_add/ps_sub = float, sin contraer), misma descuantizacion
// por los GQR vigentes, mismos bytes al FIFO y el mismo estado final de los registros que tocaba el original (r5, r6,
// r8, r9, r12, CTR, cr1, cr2, f0-f9). Al contador de ciclos se le descuenta lo que habria tardado el original.
//
// Modos (debug.stv.gc.skin al arrancar el juego; si no, [Core] STVSkinHLE): 1 = reemplazo; 2 = CONTROL: calcula pero
// deja correr el original, y GPFifo compara byte a byte lo que manda el original contra lo calculado (hay que apagar
// los atajos del FIFO: debug.stv.gc.fiforeg=0 y debug.stv.gc.fiforut=0, para que el original escriba por GPFifo).
// La firma del codigo original se verifica al compilar (JitArm64::HandleFunctionHooking): otra revision del disco no
// se toca.

#pragma clang fp contract(off)

#include "Core/HLE/HLE_STV.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#include <arm_neon.h>

#include <sys/system_properties.h>

#include "Common/Swap.h"
#include "Core/Core.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/Memmap.h"
#include "Core/PowerPC/MMU.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"

namespace HLE_STV
{
std::atomic<u32> g_skin_llamadas{0}, g_skin_vertices{0}, g_skin_comparados{0}, g_skin_difieren{0};
std::atomic<u32> g_skin_primera_dif{0};
std::atomic<u32> g_skin_gqr4{0}, g_skin_gqr7{0}, g_skin_directas{0};
static std::vector<u8> s_esperado;
static size_t s_pos = 0;
bool g_skin_control_activo = false;

// Las 87 instrucciones originales (0x8027d508 .. 0x8027d660).
static constexpr u32 FIRMA[] = {
    0xa1030020, 0x3d80cc00, 0x618c8000, 0x7d0903a6, 0x89050000, 0x10421028, 0xe0857004, 0x10631828,
    0xe0064000, 0x28880001, 0x7d2540ae, 0x29080002, 0x55293032, 0xe026c004, 0x7d044a14, 0xe0a80000,
    0x38c60006, 0xe0c80008, 0x418600b8, 0x418a0060, 0x11242116, 0x89250002, 0x11214828, 0x55293032,
    0x10e50018, 0xe0a80010, 0x11060018, 0xe0c80018, 0x10e5381e, 0xe0a80020, 0x1106401e, 0xe0c80028,
    0x10e5385c, 0xe0a80030, 0x1106405c, 0xe0c80038, 0x10e5382a, 0x7d044a14, 0x1106402a, 0x10681a5e,
    0xe0a80000, 0x1047125e, 0xe0c80008, 0x89250001, 0x10e50018, 0xe0a80010, 0x11060018, 0xe0c80018,
    0x10e5381e, 0xe0a80020, 0x1106401e, 0xe0c80028, 0x10e5385c, 0xe0a80030, 0x1106405c, 0xe0c80038,
    0x10e5382a, 0x55293032, 0x1106402a, 0x7d044a14, 0x1047111e, 0xe0a80000, 0x1068191e, 0xe0c80008,
    0x38a50008, 0x10e50018, 0xe0a80010, 0x11060018, 0xe0c80018, 0x10e5381e, 0xe0a80020, 0x1106401e,
    0xe0c80028, 0x10e5385c, 0xe0a80030, 0x1106405c, 0xe0c80038, 0x10e5382a, 0x7c002a2c, 0x1106402a,
    0x1047111c, 0x89050000, 0x1068191c, 0xf04c0000, 0xd06c0000, 0x4200fec0, 0x4e800020};

bool FirmaSkinOK(Core::System& system)
{
  auto& memory = system.GetMemory();
  for (size_t i = 0; i < sizeof(FIRMA) / sizeof(FIRMA[0]); i++)
  {
    if (memory.Read_U32(SKIN_DIR + static_cast<u32>(i * 4)) != FIRMA[i])
      return false;
  }
  return true;
}

namespace
{
struct Par
{
  float a, b;  // ps0, ps1
};

class Lector
{
public:
  Lector(const Core::CPUThreadGuard& guard, Core::System& system)
      : m_guard(guard), m_ram(system.GetMemory().GetRAM()),
        m_tam(system.GetMemory().GetRamSizeReal())
  {
  }
  u8 U8(u32 a) const
  {
    const u32 f = a & 0x3FFFFFFF;
    return (f < m_tam) ? m_ram[f] : PowerPC::MMU::HostRead<u8>(m_guard, a);
  }
  u16 U16(u32 a) const
  {
    const u32 f = a & 0x3FFFFFFF;
    if (f + 2 <= m_tam)
    {
      u16 v;
      std::memcpy(&v, m_ram + f, 2);
      return Common::swap16(v);
    }
    return PowerPC::MMU::HostRead<u16>(m_guard, a);
  }
  u32 U32(u32 a) const
  {
    const u32 f = a & 0x3FFFFFFF;
    if (f + 4 <= m_tam)
    {
      u32 v;
      std::memcpy(&v, m_ram + f, 4);
      return Common::swap32(v);
    }
    return PowerPC::MMU::HostRead<u32>(m_guard, a);
  }

private:
  const Core::CPUThreadGuard& m_guard;
  const u8* m_ram;
  u32 m_tam;
};

inline Par Sub(Par x, Par y) { return {x.a - y.a, x.b - y.b}; }

Par LeerF(const PowerPC::PowerPCState& ppc, int n)
{
  return {static_cast<float>(ppc.ps[n].PS0AsDouble()), static_cast<float>(ppc.ps[n].PS1AsDouble())};
}
void EscribirF(PowerPC::PowerPCState& ppc, int n, Par p)
{
  ppc.ps[n].SetPS0(static_cast<double>(p.a));
  ppc.ps[n].SetPS1(static_cast<double>(p.b));
}
u32 Cmp(u32 a, u32 b, const PowerPC::PowerPCState& ppc)
{
  return (a < b ? 8u : a > b ? 4u : 2u) | ppc.GetXER_SO();
}
u32 Bits(float f)
{
  u32 b;
  std::memcpy(&b, &f, 4);
  return b;
}
}  // namespace

// Descuantizacion de psq_l con el GQR fijo durante la llamada (tipo y escala calculados una vez).
struct Deq
{
  u32 tipo;
  float f;
  explicit Deq(u32 gqr)
  {
    tipo = (gqr >> 16) & 7;
    const int esc6 = static_cast<int>((gqr >> 24) & 0x3F);
    f = std::ldexp(1.0f, -((esc6 & 0x20) ? esc6 - 64 : esc6));
  }
  u32 Paso() const { return (tipo == 4 || tipo == 6) ? 1 : (tipo == 5 || tipo == 7) ? 2 : 4; }
  float Uno(const Lector& rd, u32 a) const
  {
    switch (tipo)
    {
    case 4: return static_cast<float>(rd.U8(a)) * f;
    case 5: return static_cast<float>(rd.U16(a)) * f;
    case 6: return static_cast<float>(static_cast<s8>(rd.U8(a))) * f;
    case 7: return static_cast<float>(static_cast<s16>(rd.U16(a))) * f;
    default:
    {
      const u32 b = rd.U32(a);
      float v;
      std::memcpy(&v, &b, 4);
      return v;
    }
    }
  }
  Par Dos(const Lector& rd, u32 a, bool w) const
  {
    return {Uno(rd, a), w ? 1.0f : Uno(rd, a + Paso())};
  }
};

static void Ejecutar(const Core::CPUThreadGuard& guard, bool control)
{
  auto& system = guard.GetSystem();
  auto& ppc = system.GetPPCState();
  auto& gp = system.GetGPFifo();
  const Lector rd(guard, system);
  const Deq q4(GQR(ppc, 4)), q7(GQR(ppc, 7)), q0(GQR(ppc, 0));
  const u8* ram = system.GetMemory().GetRAM();
  const u32 tam = system.GetMemory().GetRamSizeReal();

  u32 r3 = ppc.gpr[3], r4 = ppc.gpr[4], r5 = ppc.gpr[5], r6 = ppc.gpr[6];
  Par f0 = LeerF(ppc, 0), f1 = LeerF(ppc, 1), f2 = LeerF(ppc, 2), f3 = LeerF(ppc, 3), f4 = LeerF(ppc, 4),
      f5 = LeerF(ppc, 5), f6 = LeerF(ppc, 6), f7 = LeerF(ppc, 7), f8 = LeerF(ppc, 8), f9 = LeerF(ppc, 9);
  u32 r8 = rd.U16(r3 + 0x20);
  u32 ctr = r8;
  const u32 r12 = 0xCC008000;
  r8 = rd.U8(r5);
  u32 r9 = ppc.gpr[9];
  u32 cr1 = ppc.cr.GetField(1), cr2 = ppc.cr.GetField(2);
  s64 ciclos = 5 + 1;  // entrada y blr
  u32 nvert = 0;
  if (control)
  {
    s_esperado.clear();
    s_pos = 0;
  }

  // Matriz del hueso: 16 floats (8 pares). Rapido si esta en la RAM y el GQR0 es float; si no, lector general.
  float mat[16];
  auto cargar = [&](u32 m) {
    const u32 fis = m & 0x3FFFFFFF;
    if (q0.tipo == 0 && fis + 64 <= tam)
    {
      const uint8x16_t* src = reinterpret_cast<const uint8x16_t*>(ram + fis);
      for (int k = 0; k < 4; k++)
        vst1q_f32(mat + 4 * k, vreinterpretq_f32_u8(vrev32q_u8(vld1q_u8(reinterpret_cast<const u8*>(src + k)))));
    }
    else
    {
      for (int k = 0; k < 8; k++)
      {
        const Par p = q0.Dos(rd, m + 8 * k, false);
        mat[2 * k] = p.a;
        mat[2 * k + 1] = p.b;
      }
    }
  };
  // f7 = (fila0..1)*x, fma (fila 4..5)*y, fma (8..9)*z, + (12..13); f8 lo mismo con 2..3, 6..7, 10..11, 14..15:
  // un vector de 4 carriles (f7.a, f7.b, f8.a, f8.b) con las mismas operaciones por carril.
  auto transformar = [&]() -> float32x4_t {
    float32x4_t v = vmulq_n_f32(vld1q_f32(mat + 0), f0.a);
    v = vfmaq_n_f32(v, vld1q_f32(mat + 4), f0.b);
    v = vfmaq_n_f32(v, vld1q_f32(mat + 8), f1.a);
    return vaddq_f32(vld1q_f32(mat + 12), v);
  };
  float32x4_t acc;
  // Caso del juego (medido: GQR7 = 0x0f050f05, pesos u16 con escala 15; GQR4 = 0x0d070d07, posiciones s16 con escala
  // 13; flujos enteros en la RAM): lectura directa.
  const u32 fr5 = r5 & 0x3FFFFFFF, fr6 = r6 & 0x3FFFFFFF;
  const bool directo = q7.tipo == 5 && q4.tipo == 7 && fr5 + 8ull * ctr + 8 <= tam && fr6 + 6ull * ctr <= tam;
  g_skin_gqr4.store(GQR(ppc, 4), std::memory_order_relaxed);
  g_skin_gqr7.store(GQR(ppc, 7), std::memory_order_relaxed);
  if (directo)
    g_skin_directas.fetch_add(1, std::memory_order_relaxed);
  auto pos16 = [&](u32 a) -> float {
    const u8* b = ram + (a & 0x3FFFFFFF);
    return static_cast<float>(static_cast<s16>((b[0] << 8) | b[1])) * q4.f;
  };

  do
  {
    // 8027d51c
    f2 = Sub(f2, f2);
    if (directo)
    {
      const u8* w = ram + (r5 & 0x3FFFFFFF);
      f4 = {static_cast<float>((w[4] << 8) | w[5]) * q7.f, static_cast<float>((w[6] << 8) | w[7]) * q7.f};
      f0 = {pos16(r6), pos16(r6 + 2)};
    }
    else
    {
      f4 = q7.Dos(rd, r5 + 4, false);
      f0 = q4.Dos(rd, r6 + 0, false);
    }
    f3 = Sub(f3, f3);
    acc = (float32x4_t){f2.a, f2.b, f3.a, f3.b};
    cr1 = Cmp(r8, 1, ppc);
    r9 = directo ? ram[(r5 + r8) & 0x3FFFFFFF] : rd.U8(r5 + r8);
    cr2 = Cmp(r8, 2, ppc);
    r9 <<= 6;
    f1 = directo ? Par{pos16(r6 + 4), 1.0f} : q4.Dos(rd, r6 + 4, true);
    u32 m = r4 + r9;
    r6 += 6;
    const int camino = (cr1 & 2) ? 1 : (cr2 & 2) ? 2 : 3;
    ciclos += (camino == 1) ? 14 : 15;
    if (camino == 3)
    {
      f9 = {f4.a, f4.a + f4.b};  // ps_sum1 f9, f4, f4, f4
      f9 = Sub(f1, f9);
      cargar(m);
      acc = vfmaq_n_f32(acc, transformar(), f9.b);  // ps_madds1 f2/f3 con f9.ps1
      r9 = static_cast<u32>(directo ? ram[(r5 + 2) & 0x3FFFFFFF] : rd.U8(r5 + 2)) << 6;
      m = r4 + r9;
      ciclos += 23;
    }
    if (camino >= 2)
    {
      cargar(m);
      acc = vfmaq_n_f32(acc, transformar(), f4.b);  // ps_madds1 con f4.ps1
      r9 = static_cast<u32>(directo ? ram[(r5 + 1) & 0x3FFFFFFF] : rd.U8(r5 + 1)) << 6;
      m = r4 + r9;
      ciclos += 21;
    }
    // 8027d608
    r5 += 8;
    cargar(m);
    const float32x4_t t = transformar();
    acc = vfmaq_n_f32(acc, t, f4.a);  // ps_madds0 con f4.ps0
    r8 = directo ? ram[r5 & 0x3FFFFFFF] : rd.U8(r5);
    f7 = {vgetq_lane_f32(t, 0), vgetq_lane_f32(t, 1)};
    f8 = {vgetq_lane_f32(t, 2), vgetq_lane_f32(t, 3)};
    f5 = {mat[12], mat[13]};
    f6 = {mat[14], mat[15]};
    f2 = {vgetq_lane_f32(acc, 0), vgetq_lane_f32(acc, 1)};
    f3 = {vgetq_lane_f32(acc, 2), vgetq_lane_f32(acc, 3)};
    // psq_st f2, 0(r12), 0, 0 (GQR0: floats) y stfs f3
    const u32 bx = Bits(f2.a), by = Bits(f2.b), bz = Bits(f3.a);
    if (control)
    {
      // control negativo (instrumento mudo): debug.stv.gc.skinmal=1 altera el bit mas bajo de z en lo esperado
      static const bool mal = [] {
        char v[PROP_VALUE_MAX] = {};
        return __system_property_get("debug.stv.gc.skinmal", v) > 0 && v[0] == '1';
      }();
      for (u32 b : {bx, by, mal ? (bz ^ 1u) : bz})
        for (int k = 3; k >= 0; k--)
          s_esperado.push_back(static_cast<u8>(b >> (k * 8)));
    }
    else
    {
      gp.FastWrite64((static_cast<u64>(bx) << 32) | by);
      gp.FastWrite32(bz);
      gp.FastCheckGatherPipe();
    }
    ciclos += 22;
    nvert++;
    ctr--;
  } while (ctr != 0);

  g_skin_llamadas.fetch_add(1, std::memory_order_relaxed);
  g_skin_vertices.fetch_add(nvert, std::memory_order_relaxed);
  if (control)
  {
    g_skin_control_activo = !s_esperado.empty();
    return;  // corre el original (HookType::Start)
  }
  ppc.gpr[5] = r5;
  ppc.gpr[6] = r6;
  ppc.gpr[8] = r8;
  ppc.gpr[9] = r9;
  ppc.gpr[12] = r12;
  CTR(ppc) = 0;
  ppc.cr.SetField(1, cr1);
  ppc.cr.SetField(2, cr2);
  EscribirF(ppc, 0, f0);
  EscribirF(ppc, 1, f1);
  EscribirF(ppc, 2, f2);
  EscribirF(ppc, 3, f3);
  EscribirF(ppc, 4, f4);
  EscribirF(ppc, 5, f5);
  EscribirF(ppc, 6, f6);
  EscribirF(ppc, 7, f7);
  EscribirF(ppc, 8, f8);
  EscribirF(ppc, 9, f9);
  ppc.downcount -= static_cast<int>(ciclos);
  ppc.npc = LR(ppc);
}

void SkinReemplazo(const Core::CPUThreadGuard& guard)
{
  Ejecutar(guard, false);
}

void SkinControl(const Core::CPUThreadGuard& guard)
{
  Ejecutar(guard, true);
}

// Llamado por GPFifo en cada escritura lenta mientras el control esta activo: compara contra lo calculado.
void ControlComparar(const u8* bytes, size_t n)
{
  for (size_t i = 0; i < n && s_pos < s_esperado.size(); i++, s_pos++)
  {
    g_skin_comparados.fetch_add(1, std::memory_order_relaxed);
    if (bytes[i] != s_esperado[s_pos])
    {
      if (g_skin_difieren.fetch_add(1, std::memory_order_relaxed) == 0)
        g_skin_primera_dif.store(static_cast<u32>(s_pos), std::memory_order_relaxed);
    }
  }
  if (s_pos >= s_esperado.size())
    g_skin_control_activo = false;
}
}  // namespace HLE_STV
