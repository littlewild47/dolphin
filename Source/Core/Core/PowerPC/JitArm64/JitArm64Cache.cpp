// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/JitArm64/JitArm64Cache.h"

#include "Common/CommonTypes.h"
#include "Core/PowerPC/JitArm64/Jit.h"
#include "Core/PowerPC/JitCommon/JitBase.h"

using namespace Arm64Gen;

// STV_GC_JITPF_x: prendido por defecto (ver JitArm64Cache.h).
bool g_stv_jit_pf = true;

// STV_GC_JITPF_x: registro temporal del prefetch. X17 (IP1) es de los guardados por el llamador y
// esta muerto al entrar a cualquier bloque: en normalEntry el JIT puede emitir ABI_CallFunction
// (perfilado) y el chequeo de GQR usa W0 sin guardar nada, y gpr.Start() arranca sin registros del
// juego cargados en el anfitrion. ADRP y PRFM no tocan las banderas NZCV ni la memoria.
static constexpr ARM64Reg STV_PF_REG = ARM64Reg::X17;

void JitArm64BlockCache::StvPfEscribir(ARM64XEmitter& emit, const u8* destino)
{
  const u8* pc = emit.GetCodePtr();
  if (destino)
  {
    // Primera linea de 64 B del destino: su offset dentro de la pagina es multiplo de 64, asi que
    // entra en el inmediato de PRFM (escalado por 8) sin ADD. La segunda linea es +64 (<= 4096).
    const u64 linea = reinterpret_cast<u64>(destino) & ~u64{63};
    const s64 dpag = static_cast<s64>(linea & ~u64{0xFFF}) -
                     static_cast<s64>(reinterpret_cast<u64>(pc) & ~u64{0xFFF});
    // ADRP alcanza +-4 GB (21 bits de paginas con signo).
    if (dpag >= -(s64{1} << 32) && dpag < (s64{1} << 32))
    {
      const u32 lo = static_cast<u32>(linea & 0xFFF);
      emit.ADRP(STV_PF_REG, dpag);
      emit.PRFMUnsigned(ARM64Reg::PLDL2KEEP, STV_PF_REG, lo);
      emit.PRFMUnsigned(ARM64Reg::PLDL2KEEP, STV_PF_REG, lo + 64);
      DEBUG_ASSERT(emit.GetCodePtr() == pc + STV_PF_SUBHUECO || emit.HasWriteFailed());
      return;
    }
  }
  // Sin destino: se saltea el sub-hueco (1 instruccion ejecutada en vez de 3).
  emit.B(pc + STV_PF_SUBHUECO);
  emit.NOP();
  emit.NOP();
  DEBUG_ASSERT(emit.GetCodePtr() == pc + STV_PF_SUBHUECO || emit.HasWriteFailed());
}

JitArm64BlockCache::JitArm64BlockCache(JitBase& jit) : JitBaseBlockCache{jit}
{
}

void JitArm64BlockCache::Init()
{
  JitBaseBlockCache::Init();
  ClearRangesToFree();
}

void JitArm64BlockCache::WriteLinkBlock(Arm64Gen::ARM64XEmitter& emit,
                                        const JitBlock::LinkData& source, const JitBlock* dest)
{
  const u8* start = emit.GetCodePtr();

  if (!dest)
  {
    emit.MOVI2R(DISPATCHER_PC, source.exitAddress);
    if (source.call)
    {
      if (emit.GetCodePtr() == start + BLOCK_LINK_FAST_BL_OFFSET - sizeof(u32))
        emit.NOP();
      DEBUG_ASSERT(emit.GetCodePtr() == start + BLOCK_LINK_FAST_BL_OFFSET || emit.HasWriteFailed());
      emit.BL(m_jit.GetAsmRoutines()->dispatcher);
    }
    else
    {
      emit.B(m_jit.GetAsmRoutines()->dispatcher);
    }
  }
  else
  {
    if (source.call)
    {
      // The "fast" BL should be the last instruction, so that the return address matches the
      // address that was pushed onto the stack by the function that called WriteLinkBlock
      FixupBranch fast = emit.B(CC_GT);
      emit.B(source.exitFarcode);
      DEBUG_ASSERT(emit.GetCodePtr() == start + BLOCK_LINK_FAST_BL_OFFSET || emit.HasWriteFailed());
      emit.SetJumpTarget(fast);
      emit.BL(dest->normalEntry);
    }
    else
    {
      // Are we able to jump directly to the block?
      s64 block_distance = ((s64)dest->normalEntry - (s64)emit.GetCodePtr()) >> 2;
      if (block_distance >= -0x40000 && block_distance <= 0x3FFFF)
      {
        emit.B(CC_GT, dest->normalEntry);
        emit.B(source.exitFarcode);
      }
      else
      {
        FixupBranch slow = emit.B(CC_LE);
        emit.B(dest->normalEntry);
        emit.SetJumpTarget(slow);
        emit.B(source.exitFarcode);
      }
    }
  }

  // Use a fixed number of instructions so we have enough room for any patching needed later.
  const u8* end = start + BLOCK_LINK_SIZE;
  while (emit.GetCodePtr() < end)
  {
    emit.BRK(101);
    if (emit.HasWriteFailed())
      return;
  }
  ASSERT(emit.GetCodePtr() == end);
}

void JitArm64BlockCache::WriteLinkBlock(const JitBlock::LinkData& source, const JitBlock* dest)
{
  const Common::ScopedJITPageWriteAndNoExecute enable_jit_page_writes;
  u8* location = source.exitPtrs;
  ARM64XEmitter emit(location, location + BLOCK_LINK_SIZE);

  WriteLinkBlock(emit, source, dest);
  emit.FlushIcache();

  // STV_GC_JITPF_x: al enlazar, el sub-hueco de esta salida pasa a precargar el destino. Mismo
  // contexto que la reescritura del salto de arriba (hilo de CPU, ningun bloque ejecutandose: se
  // llega desde Jit() via FinalizeBlock/LinkBlock, o desde la invalidacion), con la pagina escribible
  // y la cache de instrucciones invalidada en el rango. Al DESENLAZAR (dest == nullptr) no se toca:
  // un PRFM hacia codigo viejo o liberado es inofensivo (es una pista, nunca falla ni lee de verdad),
  // y reescribirlo podria pisar el BRK que WriteDestroyBlock deja en normalEntry del bloque fuente.
  if (dest && g_stv_jit_pf && source.stvPfHueco)
  {
    u8* hueco = source.stvPfHueco;
    ARM64XEmitter emit_pf(hueco, hueco + STV_PF_SUBHUECO);
    StvPfEscribir(emit_pf, dest->normalEntry);
    emit_pf.FlushIcache();
  }
}

void JitArm64BlockCache::WriteDestroyBlock(const JitBlock& block)
{
  // Only clear the entry point as we might still be within this block.
  ARM64XEmitter emit(block.normalEntry, block.normalEntry + 4);
  const Common::ScopedJITPageWriteAndNoExecute enable_jit_page_writes;
  emit.BRK(0x123);
  emit.FlushIcache();
}

void JitArm64BlockCache::DestroyBlock(JitBlock& block)
{
  JitBaseBlockCache::DestroyBlock(block);

  if (block.near_begin != block.near_end)
    m_ranges_to_free_on_next_codegen_near.emplace_back(block.near_begin, block.near_end);
  if (block.far_begin != block.far_end)
    m_ranges_to_free_on_next_codegen_far.emplace_back(block.far_begin, block.far_end);
}

const std::vector<std::pair<u8*, u8*>>& JitArm64BlockCache::GetRangesToFreeNear() const
{
  return m_ranges_to_free_on_next_codegen_near;
}

const std::vector<std::pair<u8*, u8*>>& JitArm64BlockCache::GetRangesToFreeFar() const
{
  return m_ranges_to_free_on_next_codegen_far;
}

void JitArm64BlockCache::ClearRangesToFree()
{
  m_ranges_to_free_on_next_codegen_near.clear();
  m_ranges_to_free_on_next_codegen_far.clear();
}
