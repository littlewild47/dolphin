// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "Common/BitSet.h"
#include "Common/CommonTypes.h"
#include "Common/RangeSet.h"
#include "Core/PowerPC/PPCTables.h"

class PPCSymbolDB;

namespace Common
{
struct Symbol;
}

namespace Core
{
class CPUThreadGuard;
}

namespace PPCAnalyst
{
struct CodeOp  // 16B
{
  UGeckoInstruction inst;
  const GekkoOPInfo* opinfo = nullptr;
  u32 address = 0;
  u32 branchTo = 0;  // if UINT32_MAX, not a branch
  BitSet32 regsIn;
  BitSet32 regsOut;
  BitSet32 fregsIn;
  s8 fregOut = 0;
  BitSet8 crIn;
  BitSet8 crOut;
  bool branchUsesCtr = false;
  bool branchIsIdleLoop = false;
  BitSet8 wantsCR;
  bool wantsFPRF = false;
  bool wantsCA = false;
  bool wantsCAInFlags = false;
  BitSet8 outputCR;
  bool outputFPRF = false;
  bool outputCA = false;
  bool canEndBlock = false;
  bool canCauseException = false;
  bool skipLRStack = false;
  bool skip = false;  // followed BL-s for example
  // STV_GC_SALTOINT_x: bc condicional hacia adelante dentro del bloque: indice de la instruccion destino (-1: no).
  s16 stvSaltoA = -1;
  // STV_GC_SALTOINT_x: destino de un salto interno (etiqueta): no se reordena ni se fusiona a traves de ella.
  bool stvEtiqueta = false;
  BitSet8 crWillBeRead;
  BitSet8 crWillBeWritten;
  BitSet8 crDiscardable;
  // which registers are still needed after this instruction in this block
  BitSet32 gprWillBeRead;
  BitSet32 gprWillBeWritten;
  BitSet32 fprWillBeRead;
  BitSet32 fprWillBeWritten;
  // which registers have values which are known to be unused after this instruction
  BitSet32 gprDiscardable;
  BitSet32 fprDiscardable;
  // we do double stores from GPRs, so we don't want to load a PowerPC floating point register into
  // an XMM only to move it again to a GPR afterwards.
  BitSet32 fprInXmm;
  // whether an fpr is known to be an actual single-precision value at this point in the block.
  BitSet32 fprIsSingle;
  // whether an fpr is known to have identical top and bottom halves (e.g. due to a single
  // instruction)
  BitSet32 fprIsDuplicated;
  // whether an fpr is the output of a single-precision arithmetic instruction, i.e. whether we can
  // convert between single and double formats by just using the host machine's instruction for it.
  // (The reason why we can't always do this is because some games rely on the exact bits of
  // denormals and SNaNs being preserved as long as no arithmetic operation is performed on them.)
  BitSet32 fprIsStoreSafeBeforeInst;
  BitSet32 fprIsStoreSafeAfterInst;

  BitSet32 GetFregsOut() const
  {
    BitSet32 result;

    if (fregOut >= 0)
      result[fregOut] = true;

    return result;
  }
};

struct BlockStats
{
  u32 numCycles;
};

struct BlockRegStats
{
  bool any;
};

using CodeBuffer = std::vector<CodeOp>;

struct CodeBlock
{
  // Beginning PPC address.
  u32 m_address = 0;

  // Number of instructions
  // Gives us the size of the block.
  u32 m_num_instructions = 0;

  // Some basic statistics about the block.
  BlockStats* m_stats = nullptr;

  // Register statistics about the block.
  BlockRegStats* m_gpa = nullptr;
  BlockRegStats* m_fpa = nullptr;

  // Are we a broken block?
  bool m_broken = false;

  // Did we have a memory_exception?
  bool m_memory_exception = false;

  // Which GQRs this block uses, if any.
  BitSet8 m_gqr_used;

  // Which GQRs this block modifies, if any.
  BitSet8 m_gqr_modified;

  // Which GPRs this block reads from before defining, if any.
  BitSet32 m_gpr_inputs;

  // Which memory locations are occupied by this block.
  Common::RangeSet<u32> m_physical_addresses;
};

class PPCAnalyzer
{
public:
  enum AnalystOption
  {
    // Conditional branch continuing
    // If the JIT core supports conditional branches within the blocks
    // Block will end on unconditional branch or other ENDBLOCK flagged instruction.
    // Requires JIT support to be enabled.
    OPTION_CONDITIONAL_CONTINUE = (1 << 0),

    // Try to inline unconditional branches/calls/returns.
    // Also track the LR value to follow unconditional return instructions.
    // Might require JIT intervention to support it correctly.
    // Especially if the BLR optimization is used.
    OPTION_BRANCH_FOLLOW = (1 << 1),

    // Complex blocks support jumping backwards on to themselves.
    // Happens commonly in loops, pretty complex to support.
    // May require register caches to use register usage metrics.
    // XXX: NOT COMPLETE
    OPTION_COMPLEX_BLOCK = (1 << 2),

    // Similar to complex blocks.
    // Instead of jumping backwards, this jumps forwards within the block.
    // Requires JIT support to work.
    // XXX: NOT COMPLETE
    OPTION_FORWARD_JUMP = (1 << 3),

    // Reorder compare/Rc instructions next to their associated branches and
    // merge in the JIT (for common cases, anyway).
    OPTION_BRANCH_MERGE = (1 << 4),

    // Reorder carry instructions next to their associated branches and pass
    // carry flags in the x86 flags between them, instead of in XER.
    OPTION_CARRY_MERGE = (1 << 5),

    // Reorder cror instructions next to their associated fcmp.
    OPTION_CROR_MERGE = (1 << 6),

    // STV_GC_VOLATIL_x: sin memcheck una carga o un guardado no sale del bloque: no corta el analisis de registros
    // descartables.
    OPTION_STV_LSU_SIN_EXCEPCION = (1 << 29),
    // STV_GC_SALTOINT_x: un bc condicional hacia una instruccion posterior del mismo bloque (direcciones contiguas)
    // salta adentro en vez de salir: el destino no se compila de nuevo como bloque aparte.
    OPTION_STV_SALTO_INTERNO = (1 << 28),
    // STV_GC_CRVOL_x: con OPTION_STV_VOLATILES_MUERTOS, tambien los campos volatiles del CR (0, 1, 5, 6, 7).
    OPTION_STV_CR_VOLATIL = (1 << 27),
    // STV_GC_HOJA_x: sin seguir ramas en general, seguir igual un bl a una funcion hoja chica en linea recta (hasta 8
    // instrucciones terminadas en blr, sin otras ramas ni LR/MSR): _savegpr/_restgpr y lectores.
    OPTION_STV_HOJA = (1 << 26),
    // STV_GC_VOLATIL_x: en un bloque que termina en blr incondicional, los registros volatiles del ABI (r0, r5-r12,
    // f0, f5-f13) estan muertos al salir.
    OPTION_STV_VOLATILES_MUERTOS = (1 << 30),
  };

  // Option setting/getting
  void SetOption(AnalystOption option) { m_options |= option; }
  void ClearOption(AnalystOption option) { m_options &= ~(option); }
  bool HasOption(AnalystOption option) const { return !!(m_options & option); }
  void SetDebuggingEnabled(bool enabled) { m_is_debugging_enabled = enabled; }
  void SetBranchFollowingEnabled(bool enabled) { m_enable_branch_following = enabled; }
  void SetFloatExceptionsEnabled(bool enabled) { m_enable_float_exceptions = enabled; }
  void SetDivByZeroExceptionsEnabled(bool enabled) { m_enable_div_by_zero_exceptions = enabled; }
  u32 Analyze(u32 address, CodeBlock* block, CodeBuffer* buffer, std::size_t block_size) const;

private:
  enum class ReorderType
  {
    Carry,
    CMP,
    CROR
  };

  bool CanSwapAdjacentOps(const CodeOp& a, const CodeOp& b) const;
  void ReorderInstructionsCore(u32 instructions, CodeOp* code, bool reverse,
                               ReorderType type) const;
  void ReorderInstructions(u32 instructions, CodeOp* code) const;
  void SetInstructionStats(CodeBlock* block, CodeOp* code, const GekkoOPInfo* opinfo) const;
  bool IsBusyWaitLoop(CodeBlock* block, CodeOp* code, size_t instructions) const;

  // Options
  u32 m_options = 0;

  bool m_is_debugging_enabled = false;
  bool m_enable_branch_following = false;
  bool m_enable_float_exceptions = false;
  bool m_enable_div_by_zero_exceptions = false;
};

void FindFunctions(const Core::CPUThreadGuard& guard, u32 startAddr, u32 endAddr,
                   PPCSymbolDB* func_db);
bool AnalyzeFunction(const Core::CPUThreadGuard& guard, u32 startAddr, Common::Symbol& func,
                     u32 max_size = 0);
bool ReanalyzeFunction(const Core::CPUThreadGuard& guard, u32 start_addr, Common::Symbol& func,
                       u32 max_size = 0);

}  // namespace PPCAnalyst
