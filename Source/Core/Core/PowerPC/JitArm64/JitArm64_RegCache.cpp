// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/JitArm64/JitArm64_RegCache.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "Common/Assert.h"
#include "Common/BitSet.h"
#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"
#include "Core/PowerPC/JitArm64/Jit.h"

using namespace Arm64Gen;

void Arm64RegCache::Init(JitArm64* jit)
{
  m_jit = jit;
  m_emit = jit;
  m_float_emit.reset(new ARM64FloatEmitter(m_emit));
  GetAllocationOrder();
}

void Arm64RegCache::DiscardRegisters(BitSet32 regs)
{
  for (int i : regs)
    DiscardRegister(i);
}

void Arm64RegCache::ResetRegisters(BitSet32 regs)
{
  for (int i : regs)
  {
    // STV_GC_JITGF_v1: un GPR fijo no vuelve a "solo en ppcState": quien llama (FallBackToInterpreter) ya recargo
    // los fijos desde ppcState despues del interprete; aca solo se marca el estado.
    if (STVEsFijo(i))
    {
      STVFijoEnHost(i);
      continue;
    }

    OpArg& reg = m_guest_registers[i];
    ARM64Reg host_reg = reg.GetReg();

    ASSERT_MSG(DYNA_REC, host_reg == ARM64Reg::INVALID_REG,
               "Attempted to reset a loaded register (did you mean to flush it?)");
    reg.Flush();
  }
}

ARM64Reg Arm64RegCache::GetReg()
{
  // If we have no registers left, dump the most stale register first
  if (GetUnlockedRegisterCount() == 0)
    FlushMostStaleRegister();

  for (auto& it : m_host_registers)
  {
    if (!it.IsLocked())
    {
      it.Lock();
      return it.GetReg();
    }
  }
  // Holy cow, how did you run out of registers?
  // We can't return anything reasonable in this case. Return INVALID_REG and watch the failure
  // happen
  ASSERT_MSG(DYNA_REC, 0, "All available registers are locked!");
  return ARM64Reg::INVALID_REG;
}

void Arm64RegCache::UpdateLastUsed(BitSet32 regs_used)
{
  for (size_t i = 0; i < m_guest_registers.size(); ++i)
  {
    OpArg& reg = m_guest_registers[i];
    if (i < 32 && regs_used[i])
      reg.ResetLastUsed();
    else
      reg.IncrementLastUsed();
  }
}

u32 Arm64RegCache::GetUnlockedRegisterCount() const
{
  u32 unlocked_registers = 0;
  for (const auto& it : m_host_registers)
  {
    if (!it.IsLocked())
      ++unlocked_registers;
  }
  return unlocked_registers;
}

void Arm64RegCache::LockRegister(ARM64Reg host_reg)
{
  auto reg = std::ranges::find(m_host_registers, host_reg, &HostReg::GetReg);
  ASSERT_MSG(DYNA_REC, reg != m_host_registers.end(),
             "Don't try locking a register that isn't in the cache. Reg {}",
             static_cast<int>(host_reg));
  reg->Lock();
}

void Arm64RegCache::UnlockRegister(ARM64Reg host_reg)
{
  auto reg = std::ranges::find(m_host_registers, host_reg, &HostReg::GetReg);
  ASSERT_MSG(DYNA_REC, reg != m_host_registers.end(),
             "Don't try unlocking a register that isn't in the cache. Reg {}",
             static_cast<int>(host_reg));
  reg->Unlock();
}

void Arm64RegCache::FlushMostStaleRegister()
{
  size_t most_stale_preg = 0;
  u32 most_stale_amount = 0;

  for (size_t i = 0; i < m_guest_registers.size(); ++i)
  {
    const auto& reg = m_guest_registers[i];
    const u32 last_used = reg.GetLastUsed();

    // STV_GC_JITGF_v1: los GPR fijos no estan en el pool: vaciarlos no libera nada.
    if (last_used > most_stale_amount && reg.IsInHostRegister() && !STVEsFijo(i))
    {
      most_stale_preg = i;
      most_stale_amount = last_used;
    }
  }

  FlushRegister(most_stale_preg, FlushMode::Full, ARM64Reg::INVALID_REG);
}

void Arm64RegCache::DiscardRegister(size_t preg)
{
  // STV_GC_JITGF_v1: descartar un GPR fijo no libera nada (no esta en el pool) y no ahorra ningun STR; se ignora
  // y el registro sigue "en su registro" (o como inmediato), con un valor que el analizador ya declaro muerto.
  if (STVEsFijo(preg))
    return;

  OpArg& reg = m_guest_registers[preg];
  ARM64Reg host_reg = reg.GetReg();
  if (!IsVector(host_reg))
    host_reg = EncodeRegTo32(host_reg);

  reg.Discard();
  if (host_reg != ARM64Reg::INVALID_REG)
    UnlockRegister(host_reg);
}

Arm64GPRCache::Arm64GPRCache() : Arm64RegCache(GUEST_GPR_COUNT + GUEST_CR_COUNT)
{
}

void Arm64GPRCache::Start(PPCAnalyst::BlockRegStats& stats)
{
  // STV_GC_JITGF_v1: al entrar a un bloque cada GPR fijo esta en su registro (invariante entre bloques).
  if (m_stv_gf != 0)
  {
    for (size_t i = 0; i < GUEST_GPR_COUNT; ++i)
    {
      if (STVEsFijo(i))
        STVFijoEnHost(i);
    }
  }
}

// STV_GC_JITGF_v1: el GPR fijo queda en su registro y "sincronizado" (en modo 2, tambien igual a ppcState).
void Arm64GPRCache::STVFijoEnHost(size_t index)
{
  OpArg& reg = m_guest_registers[GUEST_GPR_OFFSET + index];
  reg.Load(STVGfHost(index));
  reg.SetDirty(false);
}

// STV_GC_JITGF_v1: cambia el modo y reconstruye el orden de asignacion (con la valvula, sin W19-W24). Se llama con la
// cache vacia; todo invitado vuelve a "solo en ppcState" y Start() vuelve a enlazar los fijos en cada bloque.
void Arm64GPRCache::STVGfConfigurar(int modo)
{
  if (modo == m_stv_gf)
    return;
  m_stv_gf = modo;
  for (auto& reg : m_guest_registers)
    reg.Flush();
  m_host_registers.clear();
  GetAllocationOrder();
}

// STV_GC_JITGF_v1: vaciado de un GPR fijo. Nunca libera el registro. Modo 1: no escribe ppcState (la verdad es el
// registro); solo materializa un inmediato de la propagacion de constantes en el registro fijo. Modo 2: ademas
// escribe ppcState como siempre (write-through), para que la verificacion al entrar al bloque compare.
// Estados: en registro (IsInHostRegister), inmediato (IsImm, sin registro; el registro esta rancio salvo que
// IsInPPCState diga que ya tiene el valor), descartado (no se produce: DiscardRegister ignora los fijos).
void Arm64GPRCache::STVFlushFijo(size_t index, FlushMode mode)
{
  OpArg& reg = m_guest_registers[GUEST_GPR_OFFSET + index];
  const ARM64Reg host_reg = STVGfHost(index);
  const u32 ppc_offset = u32(PPCSTATE_OFF_GPR(index));

  if (reg.IsInHostRegister())
  {
    // STV_GC_REVDIF_x: el fijo pasa entre bloques en su registro: tiene que salir con los bytes en orden. En
    // MaintainState (solo caminos de salida, que no vuelven) el REV queda en ese camino y la marca sigue.
    if (reg.STVRev())
    {
      if (mode == FlushMode::MaintainState)
        m_emit->REV32(host_reg, host_reg);
      else
        return;  // sigue pendiente en su registro: lo resuelve DoJit antes de salir del bloque
    }
    if (m_stv_gf == 2 && !reg.IsInPPCState())
      m_emit->STR(IndexType::Unsigned, host_reg, PPC_REG, ppc_offset);
    if (mode != FlushMode::MaintainState)
      reg.SetDirty(false);
    return;
  }

  if (IsImm(index) && !reg.IsInPPCState())
  {
    // El valor logico es el inmediato: escribirlo en el registro fijo es correcto en cualquier camino (tambien en
    // MaintainState, donde el estado de la cache no cambia).
    m_emit->MOVI2R(host_reg, GetImm(index));
    if (m_stv_gf == 2)
      m_emit->STR(IndexType::Unsigned, host_reg, PPC_REG, ppc_offset);
  }

  if (mode != FlushMode::MaintainState)
    STVFijoEnHost(index);
}

// Returns if a register is set as an immediate. Only valid for guest GPRs.
bool Arm64GPRCache::IsImm(size_t preg) const
{
  return m_jit->GetConstantPropagation().HasGPR(preg);
}

// Gets the immediate that a register is set to. Only valid for guest GPRs.
u32 Arm64GPRCache::GetImm(size_t preg) const
{
  ASSERT(m_jit->GetConstantPropagation().HasGPR(preg));
  return m_jit->GetConstantPropagation().GetGPR(preg);
}

bool Arm64GPRCache::IsCallerSaved(ARM64Reg reg) const
{
  return ARM64XEmitter::CALLER_SAVED_GPRS[DecodeReg(reg)];
}

const OpArg& Arm64GPRCache::GetGuestGPROpArg(size_t preg) const
{
  ASSERT(preg < GUEST_GPR_COUNT);
  return m_guest_registers[preg];
}

Arm64GPRCache::GuestRegInfo Arm64GPRCache::GetGuestGPR(size_t preg)
{
  ASSERT(preg < GUEST_GPR_COUNT);
  return {32, PPCSTATE_OFF_GPR(preg), m_guest_registers[GUEST_GPR_OFFSET + preg]};
}

Arm64GPRCache::GuestRegInfo Arm64GPRCache::GetGuestCR(size_t preg)
{
  ASSERT(preg < GUEST_CR_COUNT);
  return {64, PPCSTATE_OFF_CR(preg), m_guest_registers[GUEST_CR_OFFSET + preg]};
}

Arm64GPRCache::GuestRegInfo Arm64GPRCache::GetGuestByIndex(size_t index)
{
  if (IsIndexGPR(index))
    return GetGuestGPR(index - GUEST_GPR_OFFSET);
  if (IsIndexCR(index))
    return GetGuestCR(index - GUEST_CR_OFFSET);
  ASSERT_MSG(DYNA_REC, false, "Invalid index for guest register");
  return GetGuestGPR(0);
}

// STV_GC_REVDIF_x
void Arm64GPRCache::STVRevResolver(size_t preg)
{
  OpArg& reg = m_guest_registers[GUEST_GPR_OFFSET + preg];
  if (!reg.STVRev())
    return;
  if (reg.IsInHostRegister())
  {
    const ARM64Reg host_reg = STVEsFijo(GUEST_GPR_OFFSET + preg) ? STVGfHost(preg) : reg.GetReg();
    m_emit->REV32(host_reg, host_reg);
  }
  reg.STVSetRev(false);
}

void Arm64GPRCache::STVRevResolverTodo()
{
  for (size_t i = 0; i < GUEST_GPR_COUNT; ++i)
    STVRevResolver(i);
}

void Arm64GPRCache::FlushRegister(size_t index, FlushMode mode, ARM64Reg tmp_reg)
{
  // STV_GC_JITGF_v1
  if (STVEsFijo(index))
  {
    STVFlushFijo(index, mode);
    return;
  }

  GuestRegInfo guest_reg = GetGuestByIndex(index);
  OpArg& reg = guest_reg.reg;
  size_t bitsize = guest_reg.bitsize;
  const bool is_gpr = IsIndexGPR(index);

  if (reg.IsInHostRegister())
  {
    ARM64Reg host_reg = reg.GetReg();
    // STV_GC_REVDIF_x
    if (reg.STVRev())
    {
      if (mode == FlushMode::MaintainState)
      {
        if (!reg.IsInPPCState())
        {
          m_emit->REV32(host_reg, host_reg);
          m_emit->STR(IndexType::Unsigned, host_reg, PPC_REG, u32(guest_reg.ppc_offset));
          m_emit->REV32(host_reg, host_reg);
        }
        return;
      }
      if (mode == FlushMode::Undirty)
        return;
      m_emit->REV32(host_reg, host_reg);
      reg.STVSetRev(false);
    }
    if (!reg.IsInPPCState())
      m_emit->STR(IndexType::Unsigned, host_reg, PPC_REG, u32(guest_reg.ppc_offset));

    if (mode == FlushMode::Full)
    {
      UnlockRegister(EncodeRegTo32(host_reg));
      reg.Flush();
    }
    else if (mode == FlushMode::Undirty)
    {
      reg.SetDirty(false);
    }
  }
  else if (is_gpr && IsImm(index - GUEST_GPR_OFFSET))
  {
    if (!reg.IsInPPCState())
    {
      const u32 imm = GetImm(index - GUEST_GPR_OFFSET);
      if (imm == 0)
      {
        m_emit->STR(IndexType::Unsigned, bitsize == 64 ? ARM64Reg::ZR : ARM64Reg::WZR, PPC_REG,
                    u32(guest_reg.ppc_offset));
      }
      else
      {
        bool allocated_tmp_reg = false;
        if (tmp_reg != ARM64Reg::INVALID_REG)
        {
          ASSERT(IsGPR(tmp_reg));
        }
        else
        {
          ASSERT_MSG(DYNA_REC, mode != FlushMode::MaintainState,
                     "Flushing immediate while maintaining state requires temporary register");
          tmp_reg = GetReg();
          allocated_tmp_reg = true;
        }

        const ARM64Reg encoded_tmp_reg = bitsize != 64 ? tmp_reg : EncodeRegTo64(tmp_reg);

        m_emit->MOVI2R(encoded_tmp_reg, imm);
        m_emit->STR(IndexType::Unsigned, encoded_tmp_reg, PPC_REG, u32(guest_reg.ppc_offset));

        if (allocated_tmp_reg)
          UnlockRegister(tmp_reg);
      }
    }

    if (mode == FlushMode::Full)
      reg.Flush();
    else if (mode == FlushMode::Undirty)
      reg.SetDirty(false);
  }
}

void Arm64GPRCache::FlushRegisters(BitSet32 regs, FlushMode mode, ARM64Reg tmp_reg,
                                   IgnoreDiscardedRegisters ignore_discarded_registers)
{
  // STV_GC_JITGF_v1: los fijos se vacian aparte (sin STP: no se emparejan) y el resto sigue como siempre, asi los
  // pares de no fijos (p. ej. r5/r6) se siguen guardando con un STP.
  if (m_stv_gf != 0)
  {
    const BitSet32 fijos = regs & BitSet32(STV_GF_MASCARA);
    for (int i : fijos)
      STVFlushFijo(i, mode);
    regs &= ~BitSet32(STV_GF_MASCARA);
  }

  for (auto iter = regs.begin(); iter != regs.end(); ++iter)
  {
    const int i = *iter;
    OpArg& reg = m_guest_registers[GUEST_GPR_OFFSET + i];
    ASSERT_MSG(DYNA_REC,
               ignore_discarded_registers != IgnoreDiscardedRegisters::No || reg.IsInPPCState() ||
                   reg.IsInHostRegister() || IsImm(i),
               "Attempted to flush discarded register");

    if (i + 1 < int(GUEST_GPR_COUNT) && regs[i + 1])
    {
      // We've got two guest registers in a row to store
      OpArg& reg1 = m_guest_registers[GUEST_GPR_OFFSET + i];
      OpArg& reg2 = m_guest_registers[GUEST_GPR_OFFSET + i + 1];
      const bool reg1_imm = IsImm(i);
      const bool reg2_imm = IsImm(i + 1);
      const bool reg1_zero = reg1_imm && GetImm(i) == 0;
      const bool reg2_zero = reg2_imm && GetImm(i + 1) == 0;
      const bool can_allocate_reg = mode != FlushMode::MaintainState;
      if (!reg1.STVRev() && !reg2.STVRev() && !reg1.IsInPPCState() && !reg2.IsInPPCState() &&
          (reg1.IsInHostRegister() || (reg1_imm && (reg1_zero || can_allocate_reg))) &&
          (reg2.IsInHostRegister() || (reg2_imm && (reg2_zero || can_allocate_reg))))
      {
        const size_t ppc_offset = GetGuestByIndex(i).ppc_offset;
        if (ppc_offset <= 252)
        {
          ARM64Reg RX1 = reg1_zero ? ARM64Reg::WZR : BindForRead(i);
          ARM64Reg RX2 = reg2_zero ? ARM64Reg::WZR : BindForRead(i + 1);
          m_emit->STP(IndexType::Signed, RX1, RX2, PPC_REG, u32(ppc_offset));
          if (mode == FlushMode::Full)
          {
            if (reg1.IsInHostRegister())
              UnlockRegister(reg1.GetReg());
            if (reg2.IsInHostRegister())
              UnlockRegister(reg2.GetReg());
            reg1.Flush();
            reg2.Flush();
          }
          else if (mode == FlushMode::Undirty)
          {
            reg1.SetDirty(false);
            reg2.SetDirty(false);
          }
          ++iter;
          continue;
        }
      }
    }

    FlushRegister(GUEST_GPR_OFFSET + i, mode, tmp_reg);
  }
}

void Arm64GPRCache::FlushCRRegisters(BitSet8 regs, FlushMode mode, ARM64Reg tmp_reg,
                                     IgnoreDiscardedRegisters ignore_discarded_registers)
{
  for (int i : regs)
  {
    OpArg& reg = m_guest_registers[GUEST_CR_OFFSET + i];
    ASSERT_MSG(DYNA_REC,
               ignore_discarded_registers != IgnoreDiscardedRegisters::No || reg.IsInPPCState() ||
                   reg.IsInHostRegister(),
               "Attempted to flush discarded register");

    FlushRegister(GUEST_CR_OFFSET + i, mode, tmp_reg);
  }
}

void Arm64GPRCache::DiscardCRRegisters(BitSet8 regs)
{
  for (int i : regs)
    DiscardRegister(GUEST_CR_OFFSET + i);
}

void Arm64GPRCache::ResetCRRegisters(BitSet8 regs)
{
  for (int i : regs)
  {
    OpArg& reg = m_guest_registers[GUEST_CR_OFFSET + i];
    ARM64Reg host_reg = reg.GetReg();

    ASSERT_MSG(DYNA_REC, host_reg == ARM64Reg::INVALID_REG,
               "Attempted to reset a loaded register (did you mean to flush it?)");
    reg.Flush();
  }
}

void Arm64GPRCache::Flush(FlushMode mode, ARM64Reg tmp_reg,
                          IgnoreDiscardedRegisters ignore_discarded_registers)
{
  FlushRegisters(BitSet32(0xFFFFFFFF), mode, tmp_reg, ignore_discarded_registers);
  FlushCRRegisters(BitSet8(0xFF), mode, tmp_reg, ignore_discarded_registers);
}

ARM64Reg Arm64GPRCache::BindForRead(size_t index)
{
  GuestRegInfo guest_reg = GetGuestByIndex(index);
  OpArg& reg = guest_reg.reg;
  size_t bitsize = guest_reg.bitsize;
  const bool is_gpr = IsIndexGPR(index);

  IncrementAllUsed();
  reg.ResetLastUsed();

  // STV_GC_JITGF_v1: el GPR fijo se lee de su registro; si es un inmediato no materializado, primero MOVI2R ahi.
  if (STVEsFijo(index))
  {
    const ARM64Reg host_reg = STVGfHost(index);
    if (!reg.IsInHostRegister())
    {
      if (IsImm(index - GUEST_GPR_OFFSET) && !reg.IsInPPCState())
        m_emit->MOVI2R(host_reg, GetImm(index - GUEST_GPR_OFFSET));
      reg.Load(host_reg);
    }
    else if (reg.STVRev())  // STV_GC_REVDIF_x: red de seguridad
    {
      m_emit->REV32(host_reg, host_reg);
      reg.STVSetRev(false);
    }
    return host_reg;
  }

  if (reg.IsInHostRegister())
  {
    // STV_GC_REVDIF_x: red de seguridad (DoJit ya resolvio lo que la instruccion lee).
    if (reg.STVRev())
    {
      m_emit->REV32(reg.GetReg(), reg.GetReg());
      reg.STVSetRev(false);
    }
    return reg.GetReg();
  }
  else if (is_gpr && IsImm(index - GUEST_GPR_OFFSET))
  {
    ARM64Reg host_reg = bitsize != 64 ? GetReg() : EncodeRegTo64(GetReg());
    m_emit->MOVI2R(host_reg, GetImm(index - GUEST_GPR_OFFSET));
    reg.Load(host_reg);
    return host_reg;
  }
  else  // Register isn't loaded at /all/
  {
    ASSERT_MSG(DYNA_REC, reg.IsInPPCState(), "Attempted to read discarded register");
    ARM64Reg host_reg = bitsize != 64 ? GetReg() : EncodeRegTo64(GetReg());
    reg.Load(host_reg);
    reg.SetDirty(false);
    m_emit->LDR(IndexType::Unsigned, host_reg, PPC_REG, u32(guest_reg.ppc_offset));
    return host_reg;
  }
}

void Arm64GPRCache::SetImmediateInternal(size_t index, u32 imm, bool dirty)
{
  GuestRegInfo guest_reg = GetGuestByIndex(index);
  OpArg& reg = guest_reg.reg;
  // STV_GC_JITGF_v1: el registro fijo no se desbloquea (no esta en el pool); queda como inmediato pendiente de
  // materializar (dirty) o ya presente en el registro (dirty = false: quien llama garantiza que lo tiene).
  if (reg.IsInHostRegister() && !STVEsFijo(index))
    UnlockRegister(EncodeRegTo32(reg.GetReg()));
  reg.Discard();
  reg.SetDirty(dirty);
  m_jit->GetConstantPropagation().SetGPR(index - GUEST_GPR_OFFSET, imm);
}

void Arm64GPRCache::BindForWrite(size_t index, bool will_read, bool will_write)
{
  GuestRegInfo guest_reg = GetGuestByIndex(index);
  OpArg& reg = guest_reg.reg;
  const size_t bitsize = guest_reg.bitsize;
  const bool is_gpr = IsIndexGPR(index);

  reg.ResetLastUsed();

  // STV_GC_REVDIF_x: si se va a leer, el REV pendiente se emite; si se pisa (y no hay vuelta atras de memcheck), la
  // marca se borra sin emitir nada.
  if (reg.STVRev())
  {
    if (!will_read && (will_write || !m_jit->jo.memcheck))
    {
      reg.STVSetRev(false);
    }
    else if (reg.IsInHostRegister())
    {
      const ARM64Reg h = STVEsFijo(index) ? STVGfHost(index) : reg.GetReg();
      m_emit->REV32(h, h);
      reg.STVSetRev(false);
    }
  }

  // STV_GC_JITGF_v1: el GPR fijo se escribe en su registro. Nunca hace falta LDR: el valor ya esta ahi.
  if (STVEsFijo(index))
  {
    if (!reg.IsInHostRegister())
    {
      const ARM64Reg host_reg = STVGfHost(index);
      if (IsImm(index - GUEST_GPR_OFFSET) && !reg.IsInPPCState() && (will_read || !will_write))
        m_emit->MOVI2R(host_reg, GetImm(index - GUEST_GPR_OFFSET));
      reg.Load(host_reg);
    }
    if (will_write)
    {
      reg.SetDirty(true);
      m_jit->GetConstantPropagation().ClearGPR(index - GUEST_GPR_OFFSET);
    }
    return;
  }

  if (!reg.IsInHostRegister())
  {
    if (is_gpr && IsImm(index - GUEST_GPR_OFFSET))
    {
      const ARM64Reg host_reg = bitsize != 64 ? GetReg() : EncodeRegTo64(GetReg());
      if (will_read || !will_write)
      {
        // TODO: Emitting this instruction when (!will_read && !will_write) would be unnecessary if
        // we had some way to indicate to Flush that the immediate value should be written to
        // ppcState even though there is a host register allocated
        m_emit->MOVI2R(host_reg, GetImm(index - GUEST_GPR_OFFSET));
      }
      reg.Load(host_reg);
    }
    else
    {
      ASSERT_MSG(DYNA_REC, !will_read || reg.IsInPPCState(), "Attempted to load a discarded value");
      const ARM64Reg host_reg = bitsize != 64 ? GetReg() : EncodeRegTo64(GetReg());
      reg.Load(host_reg);
      reg.SetDirty(will_write);
      if (will_read)
        m_emit->LDR(IndexType::Unsigned, host_reg, PPC_REG, u32(guest_reg.ppc_offset));
      return;
    }
  }

  if (will_write)
  {
    reg.SetDirty(true);
    if (is_gpr)
      m_jit->GetConstantPropagation().ClearGPR(index - GUEST_GPR_OFFSET);
  }
}

void Arm64GPRCache::GetAllocationOrder()
{
  // Callee saved registers first in hopes that we will keep everything stored there first
  static constexpr auto allocation_order = {
      // Callee saved
      // STV_GC_JITDC_v1: W27 es DOWNCOUNT_REG, fuera del pool.
      ARM64Reg::W26,
      ARM64Reg::W25,
      ARM64Reg::W24,
      ARM64Reg::W23,
      ARM64Reg::W22,
      ARM64Reg::W21,
      ARM64Reg::W20,
      ARM64Reg::W19,

      // Caller saved
      ARM64Reg::W17,
      ARM64Reg::W16,
      ARM64Reg::W15,
      ARM64Reg::W14,
      ARM64Reg::W13,
      ARM64Reg::W12,
      ARM64Reg::W11,
      ARM64Reg::W10,
      ARM64Reg::W9,
      ARM64Reg::W8,
      ARM64Reg::W7,
      ARM64Reg::W6,
      ARM64Reg::W5,
      ARM64Reg::W4,
      ARM64Reg::W3,
      ARM64Reg::W2,
      ARM64Reg::W1,
      ARM64Reg::W0,
      ARM64Reg::W30,
  };

  for (ARM64Reg reg : allocation_order)
  {
    // STV_GC_JITGF_v1: con la valvula, W19-W24 son de los GPR fijos (pool 27 -> 21).
    if (m_stv_gf != 0 && reg >= ARM64Reg::W19 && reg <= ARM64Reg::W24)
      continue;
    m_host_registers.push_back(HostReg(reg));
  }
}

BitSet32 Arm64GPRCache::GetCallerSavedUsed() const
{
  BitSet32 registers(0);
  for (const auto& it : m_host_registers)
  {
    if (it.IsLocked() && IsCallerSaved(it.GetReg()))
      registers[DecodeReg(it.GetReg())] = true;
  }
  return registers;
}

BitSet32 Arm64GPRCache::GetDirtyGPRs() const
{
  BitSet32 registers(0);
  for (size_t i = 0; i < GUEST_GPR_COUNT; ++i)
  {
    const OpArg& arg = m_guest_registers[GUEST_GPR_OFFSET + i];
    registers[i] = !arg.IsInPPCState();
  }
  return registers;
}

void Arm64GPRCache::FlushByHost(ARM64Reg host_reg, ARM64Reg tmp_reg)
{
  for (size_t i = 0; i < m_guest_registers.size(); ++i)
  {
    const OpArg& reg = m_guest_registers[i];
    if (reg.IsInHostRegister() && DecodeReg(reg.GetReg()) == DecodeReg(host_reg))
    {
      FlushRegister(i, FlushMode::Full, tmp_reg);
      return;
    }
  }
}

// FPR Cache
constexpr size_t GUEST_FPR_COUNT = 32;

Arm64FPRCache::Arm64FPRCache() : Arm64RegCache(GUEST_FPR_COUNT)
{
}

void Arm64FPRCache::Flush(FlushMode mode, ARM64Reg tmp_reg,
                          IgnoreDiscardedRegisters ignore_discarded_registers)
{
  for (size_t i = 0; i < m_guest_registers.size(); ++i)
  {
    if (m_guest_registers[i].IsInHostRegister())
    {
      FlushRegister(i, mode, tmp_reg);
    }
    else
    {
      ASSERT_MSG(DYNA_REC,
                 ignore_discarded_registers != IgnoreDiscardedRegisters::No ||
                     m_guest_registers[i].IsInPPCState(),
                 "Attempted to flush discarded register");
    }
  }
}

ARM64Reg Arm64FPRCache::R(size_t preg, RegType type)
{
  OpArg& reg = m_guest_registers[preg];
  IncrementAllUsed();
  reg.ResetLastUsed();

  if (!reg.IsInHostRegister())
  {
    ASSERT_MSG(DYNA_REC, reg.IsInPPCState(), "Attempted to read discarded register");

    ARM64Reg host_reg = GetReg();
    u32 load_size;
    if (type == RegType::Register)
    {
      load_size = 128;
      reg.Load(host_reg, RegType::Register);
    }
    else
    {
      load_size = 64;
      reg.Load(host_reg, RegType::LowerPair);
    }
    reg.SetDirty(false);
    m_float_emit->LDR(load_size, IndexType::Unsigned, host_reg, PPC_REG,
                      static_cast<s32>(PPCSTATE_OFF_PS0(preg)));
    return host_reg;
  }

  ARM64Reg host_reg = reg.GetReg();

  switch (reg.GetFPRType())
  {
  case RegType::Single:
  {
    // We're asked for singles, so just return the register.
    if (type == RegType::Single || type == RegType::LowerPairSingle)
      return host_reg;

    // Else convert this register back to doubles.
    const ARM64Reg tmp_reg = GetReg();
    m_jit->ConvertSingleToDoublePair(preg, host_reg, host_reg, tmp_reg);
    UnlockRegister(tmp_reg);

    reg.Load(host_reg, RegType::Register);
    [[fallthrough]];
  }
  case RegType::Register:  // already in a reg
  {
    return host_reg;
  }
  case RegType::LowerPairSingle:
  {
    // We're asked for the lower single, so just return the register.
    if (type == RegType::LowerPairSingle)
      return host_reg;

    // Else convert this register back to a double.
    const ARM64Reg tmp_reg = GetReg();
    m_jit->ConvertSingleToDoubleLower(preg, host_reg, host_reg, tmp_reg);
    UnlockRegister(tmp_reg);

    reg.Load(host_reg, RegType::LowerPair);
    [[fallthrough]];
  }
  case RegType::LowerPair:
  {
    if (type == RegType::Register)
    {
      // Load the high 64bits from the file and insert them in to the high 64bits of the host
      // register
      const ARM64Reg tmp_reg = GetReg();
      m_float_emit->LDR(64, IndexType::Unsigned, tmp_reg, PPC_REG,
                        static_cast<s32>(PPCSTATE_OFF_PS1(preg)));
      m_float_emit->INS(64, host_reg, 1, tmp_reg, 0);
      UnlockRegister(tmp_reg);

      // Change it over to a full 128bit register
      reg.Load(host_reg, RegType::Register);
    }
    return host_reg;
  }
  case RegType::DuplicatedSingle:
  {
    if (type == RegType::LowerPairSingle)
      return host_reg;

    if (type == RegType::Single)
    {
      // Duplicate to the top and change over
      m_float_emit->INS(32, host_reg, 1, host_reg, 0);
      reg.Load(host_reg, RegType::Single);
      return host_reg;
    }

    const ARM64Reg tmp_reg = GetReg();
    m_jit->ConvertSingleToDoubleLower(preg, host_reg, host_reg, tmp_reg);
    UnlockRegister(tmp_reg);

    reg.Load(host_reg, RegType::Duplicated);
    [[fallthrough]];
  }
  case RegType::Duplicated:
  {
    if (type == RegType::Register)
    {
      // We are requesting a full 128bit register
      // but we are only available in the lower 64bits
      // Duplicate to the top and change over
      m_float_emit->INS(64, host_reg, 1, host_reg, 0);
      reg.Load(host_reg, RegType::Register);
    }
    return host_reg;
  }
  default:
    DEBUG_ASSERT_MSG(DYNA_REC, false, "Invalid OpArg Type!");
    break;
  }
  // We've got an issue if we end up here
  return ARM64Reg::INVALID_REG;
}

ARM64Reg Arm64FPRCache::RW(size_t preg, RegType type, bool set_dirty)
{
  OpArg& reg = m_guest_registers[preg];

  IncrementAllUsed();
  reg.ResetLastUsed();

  // If PS1 is dirty, but the caller wants a RegType with only PS0, we must write PS1 to m_ppc_state
  // now so the contents of PS1 aren't lost.
  if (!reg.IsInPPCState() && (type == RegType::LowerPair || type == RegType::LowerPairSingle))
  {
    // We must *not* modify host_reg, as the current guest instruction might want to read its old
    // value before overwriting it. So it's fine to store this register, but it's *not* fine to
    // convert it to double in place. For double conversion, a temporary register needs to be used.
    ARM64Reg host_reg = reg.GetReg();
    ARM64Reg flush_reg = host_reg;

    switch (reg.GetFPRType())
    {
    case RegType::Single:
      // For a store-safe register, conversion is just one instruction regardless of whether
      // we're whether we're converting a pair, so ConvertSingleToDoublePair followed by a
      // 128-bit store is faster than INS followed by ConvertSingleToDoubleLower and a
      // 64-bit store. But for registers which are not store-safe, the latter is better.
      flush_reg = GetReg();
      if (!m_jit->IsFPRStoreSafe(preg))
      {
        ARM64Reg scratch_reg = GetReg();
        m_float_emit->INS(32, flush_reg, 0, host_reg, 1);
        m_jit->ConvertSingleToDoubleLower(preg, flush_reg, flush_reg, scratch_reg);
        m_float_emit->STR(64, IndexType::Unsigned, flush_reg, PPC_REG, u32(PPCSTATE_OFF_PS1(preg)));
        Unlock(scratch_reg);
        reg.Load(host_reg, RegType::LowerPairSingle);
        break;
      }
      else
      {
        m_jit->ConvertSingleToDoublePair(preg, flush_reg, host_reg, flush_reg);
        m_float_emit->STR(128, IndexType::Unsigned, flush_reg, PPC_REG,
                          u32(PPCSTATE_OFF_PS0(preg)));
        reg.SetDirty(false);
      }
      break;
    case RegType::Register:
      // We are doing a full 128bit store because it takes 2 cycles on a Cortex-A57 to do a 128bit
      // store.
      // It would take longer to do an insert to a temporary and a 64bit store than to just do this.
      m_float_emit->STR(128, IndexType::Unsigned, flush_reg, PPC_REG,
                        static_cast<s32>(PPCSTATE_OFF_PS0(preg)));
      reg.SetDirty(false);
      break;
    case RegType::DuplicatedSingle:
      flush_reg = GetReg();
      m_jit->ConvertSingleToDoubleLower(preg, flush_reg, host_reg, flush_reg);
      [[fallthrough]];
    case RegType::Duplicated:
      // Store PSR1 (which is equal to PSR0) in memory.
      m_float_emit->STR(64, IndexType::Unsigned, flush_reg, PPC_REG,
                        static_cast<s32>(PPCSTATE_OFF_PS1(preg)));
      reg.Load(host_reg, reg.GetFPRType() == RegType::DuplicatedSingle ? RegType::LowerPairSingle :
                                                                         RegType::LowerPair);
      break;
    default:
      // All other types doesn't store anything in PSR1.
      break;
    }

    if (host_reg != flush_reg)
      Unlock(flush_reg);
  }

  if (!reg.IsInHostRegister())
  {
    // If not loaded at all, just alloc a new one.
    reg.Load(GetReg(), type);
    reg.SetDirty(set_dirty);
  }
  else if (set_dirty)
  {
    reg.Load(reg.GetReg(), type);
    reg.SetDirty(true);
  }

  return reg.GetReg();
}

void Arm64FPRCache::GetAllocationOrder()
{
  static constexpr auto allocation_order = {
      // Callee saved
      ARM64Reg::Q8,
      ARM64Reg::Q9,
      ARM64Reg::Q10,
      ARM64Reg::Q11,
      ARM64Reg::Q12,
      ARM64Reg::Q13,
      ARM64Reg::Q14,
      ARM64Reg::Q15,

      // Caller saved
      ARM64Reg::Q16,
      ARM64Reg::Q17,
      ARM64Reg::Q18,
      ARM64Reg::Q19,
      ARM64Reg::Q20,
      ARM64Reg::Q21,
      ARM64Reg::Q22,
      ARM64Reg::Q23,
      ARM64Reg::Q24,
      ARM64Reg::Q25,
      ARM64Reg::Q26,
      ARM64Reg::Q27,
      ARM64Reg::Q28,
      ARM64Reg::Q29,
      ARM64Reg::Q30,
      ARM64Reg::Q31,
      ARM64Reg::Q7,
      ARM64Reg::Q6,
      ARM64Reg::Q5,
      ARM64Reg::Q4,
      ARM64Reg::Q3,
      ARM64Reg::Q2,
      ARM64Reg::Q1,
      ARM64Reg::Q0,
  };

  for (ARM64Reg reg : allocation_order)
    m_host_registers.push_back(HostReg(reg));
}

void Arm64FPRCache::FlushByHost(ARM64Reg host_reg, ARM64Reg tmp_reg)
{
  for (size_t i = 0; i < m_guest_registers.size(); ++i)
  {
    const OpArg& reg = m_guest_registers[i];

    if (reg.IsInHostRegister() && reg.GetReg() == host_reg)
    {
      FlushRegister(i, FlushMode::Full, tmp_reg);
      return;
    }
  }
}

bool Arm64FPRCache::IsCallerSaved(ARM64Reg reg) const
{
  return ARM64XEmitter::CALLER_SAVED_FPRS[DecodeReg(reg)];
}

bool Arm64FPRCache::IsTopHalfUsed(ARM64Reg reg) const
{
  for (const OpArg& r : m_guest_registers)
  {
    if (r.IsInHostRegister() && DecodeReg(r.GetReg()) == DecodeReg(reg))
      return r.GetFPRType() == RegType::Register;
  }

  return false;
}

void Arm64FPRCache::FlushRegister(size_t preg, FlushMode mode, ARM64Reg tmp_reg)
{
  OpArg& reg = m_guest_registers[preg];
  const ARM64Reg host_reg = reg.GetReg();
  const bool dirty = !reg.IsInPPCState();
  RegType type = reg.GetFPRType();

  if (mode == FlushMode::Undirty)
  {
    switch (type)
    {
    case RegType::Single:
    case RegType::DuplicatedSingle:
    case RegType::LowerPairSingle:
      // In this situation, skip flushing. It's usually better to wait until later instead to avoid
      // extra conversions. We can revisit this decision in the future if the register cache gets
      // the ability to store both the single and double versions of a value simultaneously.
      return;
    default:
      break;
    }
  }

  bool allocated_tmp_reg = false;
  if (tmp_reg != ARM64Reg::INVALID_REG)
  {
    ASSERT(IsVector(tmp_reg));
  }
  else if (GetUnlockedRegisterCount() > 0)
  {
    // Calling GetReg here with 0 registers free could cause problems for two reasons:
    //
    // 1. When GetReg needs to flush, it calls this function, which can lead to infinite recursion
    // 2. When GetReg needs to flush, it does not respect mode == FlushMode::MaintainState
    //
    // So if we have 0 registers free, just don't allocate a temporary register.
    // The emitted code will still work but might be a little less efficient.

    tmp_reg = GetReg();
    allocated_tmp_reg = true;
  }

  // If we're in single mode, just convert it back to a double.
  if (type == RegType::Single)
  {
    if (dirty)
      m_jit->ConvertSingleToDoublePair(preg, host_reg, host_reg, tmp_reg);
    type = RegType::Register;
  }
  if (type == RegType::DuplicatedSingle || type == RegType::LowerPairSingle)
  {
    if (dirty)
      m_jit->ConvertSingleToDoubleLower(preg, host_reg, host_reg, tmp_reg);

    if (type == RegType::DuplicatedSingle)
      type = RegType::Duplicated;
    else
      type = RegType::LowerPair;
  }

  if (type == RegType::Register || type == RegType::LowerPair)
  {
    u32 store_size;
    if (type == RegType::Register)
      store_size = 128;
    else
      store_size = 64;

    if (dirty)
    {
      m_float_emit->STR(store_size, IndexType::Unsigned, host_reg, PPC_REG,
                        static_cast<s32>(PPCSTATE_OFF_PS0(preg)));
    }

    if (mode == FlushMode::Full)
    {
      UnlockRegister(host_reg);
      reg.Flush();
    }
    else if (mode == FlushMode::Undirty)
    {
      reg.SetDirty(false);
    }
  }
  else if (type == RegType::Duplicated)
  {
    if (dirty)
    {
      if (PPCSTATE_OFF_PS0(preg) <= 504)
      {
        m_float_emit->STP(64, IndexType::Signed, host_reg, host_reg, PPC_REG,
                          static_cast<s32>(PPCSTATE_OFF_PS0(preg)));
      }
      else
      {
        m_float_emit->STR(64, IndexType::Unsigned, host_reg, PPC_REG,
                          static_cast<s32>(PPCSTATE_OFF_PS0(preg)));
        m_float_emit->STR(64, IndexType::Unsigned, host_reg, PPC_REG,
                          static_cast<s32>(PPCSTATE_OFF_PS1(preg)));
      }
    }

    if (mode == FlushMode::Full)
    {
      UnlockRegister(host_reg);
      reg.Flush();
    }
    else if (mode == FlushMode::Undirty)
    {
      reg.SetDirty(false);
    }
  }

  if (allocated_tmp_reg)
    UnlockRegister(tmp_reg);
}

void Arm64FPRCache::FlushRegisters(BitSet32 regs, FlushMode mode, ARM64Reg tmp_reg)
{
  for (int j : regs)
    FlushRegister(j, mode, tmp_reg);
}

BitSet32 Arm64FPRCache::GetCallerSavedUsed() const
{
  BitSet32 registers(0);
  for (const auto& it : m_host_registers)
  {
    if (it.IsLocked() && (IsCallerSaved(it.GetReg()) || IsTopHalfUsed(it.GetReg())))
      registers[DecodeReg(it.GetReg())] = true;
  }
  return registers;
}

bool Arm64FPRCache::IsSingle(size_t preg, bool lower_only) const
{
  const RegType type = m_guest_registers[preg].GetFPRType();
  return type == RegType::Single || type == RegType::DuplicatedSingle ||
         (lower_only && type == RegType::LowerPairSingle);
}

void Arm64FPRCache::FixSinglePrecision(size_t preg)
{
  OpArg& reg = m_guest_registers[preg];
  if (!reg.IsInHostRegister())
    return;

  ARM64Reg host_reg = reg.GetReg();
  if (reg.GetFPRType() == RegType::Duplicated)  // only PS0 needs to be converted
  {
    m_float_emit->FCVT(32, 64, EncodeRegToDouble(host_reg), EncodeRegToDouble(host_reg));
    reg.Load(host_reg, RegType::DuplicatedSingle);
  }
  else if (reg.GetFPRType() == RegType::Register)  // PS0 and PS1 need to be converted
  {
    m_float_emit->FCVTN(32, EncodeRegToDouble(host_reg), EncodeRegToDouble(host_reg));
    reg.Load(host_reg, RegType::Single);
  }
}
