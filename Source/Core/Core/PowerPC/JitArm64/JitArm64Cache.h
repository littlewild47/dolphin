// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "Common/Arm64Emitter.h"
#include "Core/PowerPC/JitCommon/JitCache.h"

class JitBase;

typedef void (*CompiledCode)();

// STV_GC_JITPF_x (EXPLORACION): prefetch a la L2 del codigo de los bloques destino. Si esta
// prendido, cada bloque nuevo reserva al entrar (normalEntry) un hueco de STV_PF_HUECO bytes, y al
// enlazar una de sus dos primeras salidas se escribe en el sub-hueco correspondiente
// ADRP X17 + 2 x PRFM PLDL2KEEP hacia las dos primeras lineas de 64 B del destino. Se lee al
// compilar y al enlazar; la valvula debug.stv.gc.jit "pf=0|1" lo cambia y vacia la cache del JIT.
extern bool g_stv_jit_pf;

class JitArm64BlockCache : public JitBaseBlockCache
{
public:
  explicit JitArm64BlockCache(JitBase& jit);

  void Init() override;

  void DestroyBlock(JitBlock& block) override;

  const std::vector<std::pair<u8*, u8*>>& GetRangesToFreeNear() const;
  const std::vector<std::pair<u8*, u8*>>& GetRangesToFreeFar() const;

  void ClearRangesToFree();

  void WriteLinkBlock(Arm64Gen::ARM64XEmitter& emit, const JitBlock::LinkData& source,
                      const JitBlock* dest = nullptr);

  static constexpr size_t BLOCK_LINK_SIZE = 3 * sizeof(u32);
  static constexpr size_t BLOCK_LINK_FAST_BL_OFFSET = BLOCK_LINK_SIZE - sizeof(u32);

  // STV_GC_JITPF_x: un sub-hueco por destino (3 instrucciones), dos destinos por bloque.
  static constexpr size_t STV_PF_DESTINOS = 2;
  static constexpr size_t STV_PF_SUBHUECO = 3 * sizeof(u32);
  static constexpr size_t STV_PF_HUECO = STV_PF_DESTINOS * STV_PF_SUBHUECO;
  // Escribe un sub-hueco en la posicion actual de emit: con destino (y a alcance de ADRP), el
  // prefetch hacia el; si no, "B +12; NOP; NOP" (lo saltea). Siempre STV_PF_SUBHUECO bytes.
  static void StvPfEscribir(Arm64Gen::ARM64XEmitter& emit, const u8* destino);

private:
  void WriteLinkBlock(const JitBlock::LinkData& source, const JitBlock* dest) override;
  void WriteDestroyBlock(const JitBlock& block) override;

  std::vector<std::pair<u8*, u8*>> m_ranges_to_free_on_next_codegen_near;
  std::vector<std::pair<u8*, u8*>> m_ranges_to_free_on_next_codegen_far;
};
