// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/JitArm64/Jit.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <utility>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "Common/Arm64Emitter.h"
#include "Common/CommonTypes.h"
#include "Common/GekkoDisassembler.h"
#include "Common/HostDisassembler.h"
#include "Common/Logging/Log.h"
#include "Common/MathUtil.h"
#include "Common/MsgHandler.h"
#include "Common/StringUtil.h"

#include "Core/ConfigManager.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/HLE/HLE.h"
#include "Core/GeckoCode.h"
#ifdef __ANDROID__
#include <android/log.h>
#endif
#include "Core/HW/CPU.h"
#include "Core/HW/GPFifo.h"
#include "Core/HW/Memmap.h"
#include "Core/HW/ProcessorInterface.h"
#include "Core/Host.h"
#include "Core/PatchEngine.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/JitArm64/JitArm64_RegCache.h"
#include "Core/PowerPC/JitCommon/ConstantPropagation.h"
#include "Core/PowerPC/JitInterface.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/System.h"

using namespace Arm64Gen;

constexpr size_t NEAR_CODE_SIZE = 1024 * 1024 * 64;
// We use a bigger farcode size for JitArm64 than Jit64, because JitArm64 always emits farcode
// for the slow path of each loadstore instruction. Jit64 postpones emitting farcode until the
// farcode actually is needed, saving it from having to emit farcode for most instructions.
// TODO: Perhaps implement something similar to Jit64. But using more RAM isn't much of a problem.
constexpr size_t FAR_CODE_SIZE = 1024 * 1024 * 64;
constexpr size_t TOTAL_CODE_SIZE = NEAR_CODE_SIZE * 2 + FAR_CODE_SIZE * 2;

JitArm64::JitArm64(Core::System& system)
    : JitBase(system), m_float_emit(this),
      m_disassembler(HostDisassembler::Factory(HostDisassembler::Platform::aarch64))
{
}

JitArm64::~JitArm64() = default;

void JitArm64::Init()
{
  InitFastmemArena();

  RefreshConfig();

  // We want the regions to be laid out in this order in memory:
  // m_far_code_0, m_near_code_0, m_near_code_1, m_far_code_1.
  // AddChildCodeSpace grabs space from the end of the parent region,
  // so we have to call AddChildCodeSpace in reverse order.
  AllocCodeSpace(TOTAL_CODE_SIZE);
  AddChildCodeSpace(&m_far_code_1, FAR_CODE_SIZE);
  AddChildCodeSpace(&m_near_code_1, NEAR_CODE_SIZE);
  AddChildCodeSpace(&m_near_code_0, NEAR_CODE_SIZE);
  AddChildCodeSpace(&m_far_code_0, FAR_CODE_SIZE);
  ASSERT(m_far_code_0.GetCodeEnd() == m_near_code_0.GetCodePtr());
  ASSERT(m_near_code_0.GetCodeEnd() == m_near_code_1.GetCodePtr());
  ASSERT(m_near_code_1.GetCodeEnd() == m_far_code_1.GetCodePtr());

  jo.optimizeGatherPipe = true;
  SetBlockLinkingEnabled(true);
  SetOptimizationEnabled(true);
  gpr.Init(this);
  fpr.Init(this);
  // STV_GC_JITGF_v1: el orden de asignacion depende de la valvula (RefreshConfig ya corrio).
  gpr.STVGfConfigurar(STVGf());
  gpr.STVBaseConfigurar((m_stv_volatil & 16384) != 0, (m_stv_volatil & 81920) == 81920);  // STV_GC_BASE_x
  blocks.Init();

  code_block.m_stats = &js.st;
  code_block.m_gpa = &js.gpa;
  code_block.m_fpa = &js.fpa;

  InitBLROptimization();

  GenerateAsmAndResetFreeMemoryRanges();
}

void JitArm64::SetBlockLinkingEnabled(bool enabled)
{
  jo.enableBlocklink = enabled && !SConfig::GetInstance().bJITNoBlockLinking;
}

void JitArm64::SetOptimizationEnabled(bool enabled)
{
  if (enabled)
  {
    analyzer.SetOption(PPCAnalyst::PPCAnalyzer::OPTION_CONDITIONAL_CONTINUE);
    analyzer.SetOption(PPCAnalyst::PPCAnalyzer::OPTION_CARRY_MERGE);
    analyzer.SetOption(PPCAnalyst::PPCAnalyzer::OPTION_BRANCH_FOLLOW);
  }
  else
  {
    analyzer.ClearOption(PPCAnalyst::PPCAnalyzer::OPTION_CONDITIONAL_CONTINUE);
    analyzer.ClearOption(PPCAnalyst::PPCAnalyzer::OPTION_CARRY_MERGE);
    analyzer.ClearOption(PPCAnalyst::PPCAnalyzer::OPTION_BRANCH_FOLLOW);
  }
}

bool JitArm64::HandleFault(uintptr_t access_address, SContext* ctx)
{
  // Ifdef this since the exception handler runs on a separate thread on macOS (ARM)
#if !(defined(__APPLE__) && defined(_M_ARM_64))
  // We can't handle any fault from other threads.
  if (!Core::IsCPUThread())
  {
    ERROR_LOG_FMT(DYNA_REC, "Exception handler - Not on CPU thread");
    DoBacktrace(access_address, ctx);
    return false;
  }
#endif

  bool success = false;

  // Handle BLR stack faults, may happen in C++ code.
  const uintptr_t stack_guard = reinterpret_cast<uintptr_t>(m_stack_guard);
  if (access_address >= stack_guard && access_address < stack_guard + GUARD_SIZE)
    success = HandleStackFault();

  // If the fault is in JIT code space, look for fastmem areas.
  if (!success && IsInSpaceOrChildSpace(reinterpret_cast<u8*>(ctx->CTX_PC)))
  {
    auto& memory = m_system.GetMemory();
    if (memory.IsAddressInFastmemArea(reinterpret_cast<u8*>(access_address)))
    {
      const uintptr_t memory_base = reinterpret_cast<uintptr_t>(
          m_ppc_state.msr.DR ? memory.GetLogicalBase() : memory.GetPhysicalBase());

      if (access_address < memory_base || access_address >= memory_base + 0x1'0000'0000)
      {
        ERROR_LOG_FMT(DYNA_REC,
                      "JitArm64 address calculation overflowed. This should never happen! "
                      "PC {:#018x}, access address {:#018x}, memory base {:#018x}, MSR.DR {}, "
                      "mem_ptr {}, pbase {}, lbase {}",
                      ctx->CTX_PC, access_address, memory_base, m_ppc_state.msr.DR,
                      fmt::ptr(m_ppc_state.mem_ptr), fmt::ptr(memory.GetPhysicalBase()),
                      fmt::ptr(memory.GetLogicalBase()));
      }
      else
      {
        success = HandleFastmemFault(ctx);
      }
    }
  }

  if (!success)
  {
    ERROR_LOG_FMT(DYNA_REC, "Exception handler - Unhandled fault");
    DoBacktrace(access_address, ctx);
  }
  return success;
}

void JitArm64::ClearCache()
{
  m_fault_to_handler.clear();

  blocks.Clear();
  blocks.ClearRangesToFree();
  const Common::ScopedJITPageWriteAndNoExecute enable_jit_page_writes;
  m_far_code_0.ClearCodeSpace();
  m_near_code_0.ClearCodeSpace();
  m_near_code_1.ClearCodeSpace();
  m_far_code_1.ClearCodeSpace();
  RefreshConfig();
  // STV_GC_JITGF_v1: aplicar el modo nuevo a la cache de registros antes de regenerar las rutinas y compilar.
  gpr.STVGfConfigurar(STVGf());
  gpr.STVBaseConfigurar((m_stv_volatil & 16384) != 0, (m_stv_volatil & 81920) == 81920);  // STV_GC_BASE_x

  GenerateAsmAndResetFreeMemoryRanges();
}

void JitArm64::GenerateAsmAndResetFreeMemoryRanges()
{
  SetCodePtr(m_near_code_1.GetWritableCodePtr(), m_near_code_1.GetWritableCodeEnd());
  m_far_code.SetCodePtr(m_far_code_1.GetWritableCodePtr(), m_far_code_1.GetWritableCodeEnd());

  const u8* routines_near_start = GetCodePtr();
  const u8* routines_far_start = m_far_code.GetCodePtr();

  GenerateAsm();

  const u8* routines_near_end = GetCodePtr();
  const u8* routines_far_end = m_far_code.GetCodePtr();

  ResetFreeMemoryRanges(routines_near_end - routines_near_start,
                        routines_far_end - routines_far_start);

  Host_JitCacheInvalidation();
}

void JitArm64::FreeRanges()
{
  // Check if any code blocks have been freed in the block cache and transfer this information to
  // the local rangesets to allow overwriting them with new code.
  for (const auto& [from, to] : blocks.GetRangesToFreeNear())
  {
    const auto first_fastmem_area = m_fault_to_handler.upper_bound(from);
    auto last_fastmem_area = first_fastmem_area;
    const auto end = m_fault_to_handler.end();
    while (last_fastmem_area != end && last_fastmem_area->first <= to)
      ++last_fastmem_area;
    m_fault_to_handler.erase(first_fastmem_area, last_fastmem_area);

    if (from < m_near_code_0.GetCodeEnd())
      m_free_ranges_near_0.insert(from, to);
    else
      m_free_ranges_near_1.insert(from, to);
  }
  for (const auto& [from, to] : blocks.GetRangesToFreeFar())
  {
    if (from < m_far_code_0.GetCodeEnd())
      m_free_ranges_far_0.insert(from, to);
    else
      m_free_ranges_far_1.insert(from, to);
  }
  blocks.ClearRangesToFree();
}

void JitArm64::ResetFreeMemoryRanges(size_t routines_near_size, size_t routines_far_size)
{
  // Set the near and far code regions as unused.
  m_free_ranges_far_0.clear();
  m_free_ranges_far_0.insert(m_far_code_0.GetWritableCodePtr() + routines_near_size,
                             m_far_code_0.GetWritableCodeEnd());
  m_free_ranges_near_0.clear();
  m_free_ranges_near_0.insert(m_near_code_0.GetWritableCodePtr(),
                              m_near_code_0.GetWritableCodeEnd());
  m_free_ranges_near_1.clear();
  m_free_ranges_near_1.insert(m_near_code_1.GetWritableCodePtr() + routines_near_size,
                              m_near_code_1.GetWritableCodeEnd());
  m_free_ranges_far_1.clear();
  m_free_ranges_far_1.insert(m_far_code_1.GetWritableCodePtr() + routines_far_size,
                             m_far_code_1.GetWritableCodeEnd());
}

void JitArm64::Shutdown()
{
  auto& memory = m_system.GetMemory();
  memory.ShutdownFastmemArena();
  FreeCodeSpace();
  blocks.Shutdown();
}

void JitArm64::FallBackToInterpreter(UGeckoInstruction inst)
{
  FlushCarry();
  gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG, IgnoreDiscardedRegisters::Yes);
  fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG, IgnoreDiscardedRegisters::Yes);

  if (js.op->canEndBlock)
  {
    // also flush the program counter
    auto WA = gpr.GetScopedReg();
    MOVI2R(WA, js.compilerPC);
    STR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(pc));
    ADD(WA, WA, 4);
    STR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(npc));
  }

  Interpreter::Instruction instr = Interpreter::GetInterpreterOp(inst);
  // STV_GC_JITGF_v1: el interprete lee y escribe ppcState.gpr: los fijos van a memoria antes y vuelven despues
  // (todos: el interprete puede leer cualquiera; el Flush de arriba ya materializo los inmediatos en su registro).
  STVGfGuardar();
  ABI_CallFunction(instr, &m_system.GetInterpreter(), inst.hex);
  STVGfCargar();

  // If the instruction wrote to any registers which were marked as discarded,
  // we must mark them as no longer discarded
  gpr.ResetRegisters(js.op->regsOut);
  fpr.ResetRegisters(js.op->GetFregsOut());
  gpr.ResetCRRegisters(js.op->crOut);

  // We must also update constant propagation
  m_constant_propagation.ClearGPRs(js.op->regsOut);

  if (js.op->opinfo->flags & FL_SET_MSR)
    EmitUpdateMembase();

  if (js.op->canEndBlock)
  {
    if (js.isLastInstruction)
    {
      auto WA = gpr.GetScopedReg();
      LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(npc));
      WriteExceptionExit(WA);
    }
    else
    {
      // only exit if ppcstate.npc was changed
      auto WA = gpr.GetScopedReg();
      LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(npc));
      {
        auto WB = gpr.GetScopedReg();
        MOVI2R(WB, js.compilerPC + 4);
        CMP(WB, WA);
      }
      FixupBranch c = B(CC_EQ);
      WriteExceptionExit(WA);
      SetJumpTarget(c);
    }
  }
  else if (ShouldHandleFPExceptionForInstruction(js.op))
  {
    WriteConditionalExceptionExit(EXCEPTION_PROGRAM);
  }

  if (jo.memcheck && (js.op->opinfo->flags & FL_LOADSTORE))
  {
    WriteConditionalExceptionExit(EXCEPTION_DSI);
  }
}

void JitArm64::HLEFunction(u32 hook_index)
{
  FlushCarry();
  gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
  fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);

  // STV_GC_JITGF_v1: las funciones HLE leen argumentos (r3...) y escriben resultados y r1 (GeckoReturnTrampoline).
  STVGfGuardar();
  ABI_CallFunction(&HLE::ExecuteFromJIT, js.compilerPC, hook_index, &m_system);
  STVGfCargar();
}

void JitArm64::DoNothing(UGeckoInstruction inst)
{
  // Yup, just don't do anything.
}

void JitArm64::Break(UGeckoInstruction inst)
{
  WARN_LOG_FMT(DYNA_REC, "Breaking! {:08x} - Fix me ;)", inst.hex);
  std::exit(0);
}

void JitArm64::Cleanup()
{
  if (jo.optimizeGatherPipe && js.fifoBytesSinceCheck > 0)
  {
    static_assert(PPCSTATE_OFF(gather_pipe_ptr) <= 504);
    static_assert(PPCSTATE_OFF(gather_pipe_ptr) + 8 == PPCSTATE_OFF(gather_pipe_base_ptr));
    LDP(IndexType::Signed, ARM64Reg::X0, ARM64Reg::X1, PPC_REG, PPCSTATE_OFF(gather_pipe_ptr));
    SUB(ARM64Reg::X0, ARM64Reg::X0, ARM64Reg::X1);
    // STV_GC_FIFO3_v1: el chequeo de fin de bloque tambien usa el umbral (era el que disparaba casi todos los vaciados:
    // FIFO2 lo dejo en 32 y por eso "el umbral no movia nada").
    CMP(ARM64Reg::X0, m_stv_gp_umbral);
    FixupBranch exit = B(CC_LT);
    ABI_CallFunction(&GPFifo::UpdateGatherPipe, &m_system.GetGPFifo());
    SetJumpTarget(exit);
  }

  if (m_ppc_state.feature_flags & FEATURE_FLAG_PERFMON)
  {
    ABI_CallFunction(&PowerPC::UpdatePerformanceMonitor, js.downcountAmount, js.numLoadStoreInst,
                     js.numFloatingPointInst, &m_ppc_state);
  }
}

void JitArm64::DoDownCount()
{
  // STV_GC_JITDC_v1: con la valvula, un solo SUBS sobre W27 (las banderas siguen alimentando el B.GT del enlace
  // y el B.LE del despachador). W1 es el temporal si el inmediato no entra en 12 bits.
  if (STVDcReg())
  {
    SUBSI2R(DOWNCOUNT_REG, DOWNCOUNT_REG, js.downcountAmount, ARM64Reg::W1);
    return;
  }
  LDR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF(downcount));
  SUBSI2R(ARM64Reg::W0, ARM64Reg::W0, js.downcountAmount, ARM64Reg::W1);
  STR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF(downcount));
}

// STV_GC_ICCUENTA_x (instrumento, bit 512 con el 128): por sitio de bcctr, llamadas, destinos distintos y aciertos que
// tendria una cache en linea de 1 y de 2 destinos (LRU). Lo escribe el CPU y lo vuelca el hilo de metricas: mutex.
namespace
{
struct STVICSitio
{
  u64 n = 0, h1 = 0, h2 = 0;
  u32 a = 0, b = 0;
  std::vector<u32> dist;
};
std::mutex s_stv_ic_mutex;
std::unordered_map<u32, STVICSitio> s_stv_ic;
}  // namespace

void STVICRegistrar(u32 sitio, u32 destino)
{
  std::lock_guard lk(s_stv_ic_mutex);
  STVICSitio& e = s_stv_ic[sitio];
  ++e.n;
  if (e.a == destino)
  {
    ++e.h1;
    ++e.h2;
  }
  else
  {
    if (e.b == destino)
      ++e.h2;
    e.b = e.a;
    e.a = destino;
  }
  if (e.dist.size() < 64 && std::ranges::find(e.dist, destino) == e.dist.end())
    e.dist.push_back(destino);
}

void STVICVolcar(const char* ruta)
{
  std::lock_guard lk(s_stv_ic_mutex);
  if (std::FILE* f = std::fopen(ruta, "w"))
  {
    for (const auto& [sitio, e] : s_stv_ic)
      std::fprintf(f, "%08x\t%llu\t%zu\t%llu\t%llu\n", sitio, (unsigned long long)e.n, e.dist.size(),
                   (unsigned long long)e.h1, (unsigned long long)e.h2);
    std::fclose(f);
  }
  s_stv_ic.clear();
}

// STV_GC_IC_x (bit 1024 con el 128): cache en linea de bcctr con aprendizaje. Un sitio sin decision llama a
// STVICAprender en cada ejecucion; con 16 destinos iguales seguidos queda ADIVINADO (el bloque se recompila con el destino
// como constante: si coincide, salida enlazada normal; si no, el despachador), con 256 llamadas sin eso queda MEGA (se
// recompila sin aprender). Al decidir se invalida el bloque del sitio y devuelve 1: ese camino descarta la pila de BLR y
// sale sin empujar marco (un retorno a un bloque destruido caeria en codigo liberado). Solo el hilo de CPU: sin mutex.
std::unordered_map<u32, STVICSitioJit> g_stv_ic_sitios;

u32 STVICAprender(u32 sitio, u32 destino)
{
  STVICSitioJit& e = g_stv_ic_sitios[sitio];
  if (e.estado != 0)
    return 0;
  ++e.llamadas;
  if (destino == e.destino)
  {
    ++e.racha;
  }
  else
  {
    e.destino = destino;
    e.racha = 1;
  }
  if (e.racha >= 16)
    e.estado = 1;
  else if (e.llamadas >= 256)
    e.estado = 2;
  else
    return 0;
  Core::System::GetInstance().GetJitInterface().InvalidateICache(sitio, 4, true);
  return 1;
}

// STV_GC_BASE_x: decide si el acceso [rA + off] va por X25 y, si hace falta, carga la base. Devuelve true si se usa.
bool JitArm64::STVBasePreparar(int guest, ARM64Reg host, s32 off, u32 bytes, bool update)
{
  if (!(m_stv_volatil & 16384) || update || !jo.fastmem || jo.memcheck || guest < 0)
    return false;
  if (js.fifoWriteAddresses.contains(js.compilerPC))
    return false;
  const bool sin_escala = off >= -256 && off <= 255;
  const bool escalado = off >= 0 && (off % bytes) == 0 && (off / bytes) <= 4095;
  if (!sin_escala && !escalado)
    return false;
  const int ranuras = (m_stv_volatil & 65536) ? 2 : 1;
  int r = -1;
  for (int k = 0; k < ranuras; k++)
  {
    if (m_stv_base_guest[k] == guest)
      r = k;
  }
  if (r < 0)
  {
    r = ranuras == 2 ? static_cast<int>(1 - m_stv_base_lru) : 0;
    const ARM64Reg x = r == 0 ? ARM64Reg::X25 : ARM64Reg::X26;
    ADD(x, MEM_REG, host, ArithOption(host, ExtendSpecifier::UXTW));
    m_stv_base_guest[r] = guest;
  }
  m_stv_base_lru = static_cast<u32>(r);
  m_stv_base_reg = r == 0 ? ARM64Reg::X25 : ARM64Reg::X26;
  m_stv_base_usar = true;
  m_stv_base_off = off;
  return true;
}

// STV_GC_SALTOINT_x / STV_GC_TALON_x: la misma rama con la condicion contraria.
FixupBranch JitArm64::STVInvertir(FixupBranch f)
{
  switch (f.type)
  {
  case FixupBranch::Type::CBZ:
    f.type = FixupBranch::Type::CBNZ;
    break;
  case FixupBranch::Type::CBNZ:
    f.type = FixupBranch::Type::CBZ;
    break;
  case FixupBranch::Type::TBZ:
    f.type = FixupBranch::Type::TBNZ;
    break;
  case FixupBranch::Type::TBNZ:
    f.type = FixupBranch::Type::TBZ;
    break;
  case FixupBranch::Type::BConditional:
    f.cond = static_cast<CCFlags>(f.cond ^ 1);
    break;
  default:
    ASSERT(false);
  }
  return f;
}

// STV_GC_SALTOINT_x: descuenta ciclos sin salir (W0/W1 de temporales: se llama con todo vaciado).
void JitArm64::STVRestarDowncount(u32 ciclos)
{
  if (STVDcReg())
  {
    SUBSI2R(DOWNCOUNT_REG, DOWNCOUNT_REG, ciclos, ARM64Reg::W1);
    return;
  }
  LDR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF(downcount));
  SUBSI2R(ARM64Reg::W0, ARM64Reg::W0, ciclos, ARM64Reg::W1);
  STR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF(downcount));
}

// STV_GC_SALTOINT_x: camino tomado de un bc interno, emitido donde iria su WriteExit. Si la rama de "no saltar" es
// una sola y el vaciado no emite nada (ni hace falta ajustar el downcount), se invierte esa rama y salta directo a la
// etiqueta: devuelve true y quien llama ya no la resuelve.
bool JitArm64::STVSaltoInterno(FixupBranch* unica, ARM64Reg tmp)
{
  const u32 destino = static_cast<u32>(js.op->stvSaltoA);
  auto it = std::ranges::find_if(m_stv_saltos, [&](const auto& p) { return p.destino == destino; });
  if (it == m_stv_saltos.end())
  {
    m_stv_saltos.push_back({destino, {}, js.downcountAmount, js.firstFPInstructionFound,
                            js.fifoBytesSinceCheck, js.mustCheckFifo});
    it = m_stv_saltos.end() - 1;
  }
  else
  {
    it->fp = it->fp && js.firstFPInstructionFound;
    it->fifo = std::max(it->fifo, js.fifoBytesSinceCheck);
    it->fifo_obligado = it->fifo_obligado || js.mustCheckFifo;
  }

  const u8* antes = GetCodePtr();
  gpr.Flush(FlushMode::MaintainState, tmp);
  fpr.Flush(FlushMode::MaintainState, ARM64Reg::INVALID_REG);
  const bool ajuste = js.downcountAmount != it->base_dc;
  if (unica && !ajuste && GetCodePtr() == antes)
  {
    FixupBranch f = STVInvertir(*unica);
    it->saltos.push_back(f);
    return true;
  }
  if (ajuste)
    STVRestarDowncount(js.downcountAmount - it->base_dc);
  it->saltos.push_back(B());
  return false;
}

// STV_GC_SALTOINT_x: llegada del camino de largo a una etiqueta con saltos pendientes.
void JitArm64::STVEtiqueta(u32 indice)
{
  auto it = std::ranges::find_if(m_stv_saltos, [&](const auto& p) { return p.destino == indice; });
  if (it == m_stv_saltos.end())
    return;
  FlushCarry();
  gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG, IgnoreDiscardedRegisters::Yes);
  fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG, IgnoreDiscardedRegisters::Yes);
  // Un descartado del camino de largo esta muerto en los dos caminos (se escribe antes de leerse); el tomado lo trae en
  // ppcState. Se toma como volcado: la union puede agregar salidas que el analisis lineal no vio (el chequeo de FPU si
  // un camino no lo hizo, el del gather pipe por el maximo) y esas vacian todo.
  gpr.STVVolatilesAlEstado();
  fpr.STVVolatilesAlEstado();
  m_constant_propagation.Clear();
  m_stv_base_guest = {-1, -1};  // STV_GC_BASE_x: el camino tomado puede no tener la base
  if (js.downcountAmount > it->base_dc)
    STVRestarDowncount(js.downcountAmount - it->base_dc);
  js.downcountAmount = it->base_dc;
  for (const FixupBranch& f : it->saltos)
    SetJumpTarget(f);
  js.firstFPInstructionFound = js.firstFPInstructionFound && it->fp;
  js.fifoBytesSinceCheck = std::max(js.fifoBytesSinceCheck, it->fifo);
  js.mustCheckFifo = js.mustCheckFifo || it->fifo_obligado;
  m_stv_saltos.erase(it);
}

// STV_GC_JITDC_v1: W27 -> ppcState.downcount. Con la valvula apagada no emite nada, o un NOP si se pide
// (rutinas del despachador: misma disposicion en los dos modos).
void JitArm64::STVDcGuardar(bool nop_si_apagada)
{
  if (STVDcReg())
    STR(IndexType::Unsigned, DOWNCOUNT_REG, PPC_REG, PPCSTATE_OFF(downcount));
  else if (nop_si_apagada)
    NOP();
}

// STV_GC_JITDC_v1: ppcState.downcount -> W27 (idem).
void JitArm64::STVDcCargar(bool nop_si_apagada)
{
  if (STVDcReg())
    LDR(IndexType::Unsigned, DOWNCOUNT_REG, PPC_REG, PPCSTATE_OFF(downcount));
  else if (nop_si_apagada)
    NOP();
}

// STV_GC_JITGF_v1: los seis GPR fijos -> ppcState.gpr (3 STP, los pares son contiguos en ppcState).
static_assert(PPCSTATE_OFF_GPR(31) <= 252, "STP/LDP de 32 bits no llega a gpr[30..31]");
void JitArm64::STVGfGuardar(bool nop_si_apagada)
{
  if (STVGf() != 0)
  {
    STP(IndexType::Signed, STVGfHost(0), STVGfHost(1), PPC_REG, PPCSTATE_OFF_GPR(0));
    STP(IndexType::Signed, STVGfHost(3), STVGfHost(4), PPC_REG, PPCSTATE_OFF_GPR(3));
    STP(IndexType::Signed, STVGfHost(30), STVGfHost(31), PPC_REG, PPCSTATE_OFF_GPR(30));
  }
  else if (nop_si_apagada)
  {
    NOP();
    NOP();
    NOP();
  }
}

// STV_GC_JITGF_v1: ppcState.gpr -> los seis GPR fijos (3 LDP).
void JitArm64::STVGfCargar(bool nop_si_apagada)
{
  if (STVGf() != 0)
  {
    LDP(IndexType::Signed, STVGfHost(0), STVGfHost(1), PPC_REG, PPCSTATE_OFF_GPR(0));
    LDP(IndexType::Signed, STVGfHost(3), STVGfHost(4), PPC_REG, PPCSTATE_OFF_GPR(3));
    LDP(IndexType::Signed, STVGfHost(30), STVGfHost(31), PPC_REG, PPCSTATE_OFF_GPR(30));
  }
  else if (nop_si_apagada)
  {
    NOP();
    NOP();
    NOP();
  }
}

// STV_GC_JITGF_v1: modo 2 (solo depuracion). Al entrar al bloque, con la escritura en ppcState de siempre activa,
// cada registro fijo tiene que ser igual a su copia en memoria. Si no, BRK #0x5A: alguien escribio ppcState.gpr de
// un fijo sin recargarlo (un C++ llamado desde un bloque, o una rutina que no paso por un punto de
// sincronizacion), o la cache de registros dejo un fijo sin actualizar (un inmediato que fue a memoria y no al
// registro). En la entrada del bloque W0, W1 y las banderas estan libres (la cache esta vacia). 12 instrucciones.
void JitArm64::STVGfVerificar()
{
  LDP(IndexType::Signed, ARM64Reg::W0, ARM64Reg::W1, PPC_REG, PPCSTATE_OFF_GPR(0));
  CMP(STVGfHost(0), ARM64Reg::W0);
  CCMP(STVGfHost(1), ARM64Reg::W1, 0, CC_EQ);
  LDP(IndexType::Signed, ARM64Reg::W0, ARM64Reg::W1, PPC_REG, PPCSTATE_OFF_GPR(3));
  CCMP(STVGfHost(3), ARM64Reg::W0, 0, CC_EQ);
  CCMP(STVGfHost(4), ARM64Reg::W1, 0, CC_EQ);
  LDP(IndexType::Signed, ARM64Reg::W0, ARM64Reg::W1, PPC_REG, PPCSTATE_OFF_GPR(30));
  CCMP(STVGfHost(30), ARM64Reg::W0, 0, CC_EQ);
  CCMP(STVGfHost(31), ARM64Reg::W1, 0, CC_EQ);
  FixupBranch iguales = B(CC_EQ);
  BRK(0x5A);
  SetJumpTarget(iguales);
}

void JitArm64::ResetStack()
{
  if (!m_enable_blr_optimization)
    return;

  LDR(IndexType::Unsigned, ARM64Reg::X0, PPC_REG, PPCSTATE_OFF(stored_stack_pointer));
  ADD(ARM64Reg::SP, ARM64Reg::X0, 0);
}

void JitArm64::IntializeSpeculativeConstants()
{
  // If the block depends on an input register which looks like a gather pipe or MMIO related
  // constant, guess that it is actually a constant input, and specialize the block based on this
  // assumption. This happens when there are branches in code writing to the gather pipe, but only
  // the first block loads the constant.
  // Insert a check at the start of the block to verify that the value is actually constant.
  // This can save a lot of backpatching and optimize gather pipe writes in more places.
  const u8* fail = nullptr;
  for (auto i : code_block.m_gpr_inputs)
  {
    u32 compile_time_value = m_ppc_state.gpr[i];
    if (m_mmu.IsOptimizableGatherPipeWrite(compile_time_value) ||
        m_mmu.IsOptimizableGatherPipeWrite(compile_time_value - 0x8000) ||
        compile_time_value == 0xCC000000)
    {
      if (!fail)
      {
        SwitchToFarCode();
        fail = GetCodePtr();
        MOVI2R(DISPATCHER_PC, js.blockStart);
        STR(IndexType::Unsigned, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(pc));
        ABI_CallFunction(&JitInterface::CompileExceptionCheckFromJIT, &m_system.GetJitInterface(),
                         static_cast<u32>(JitInterface::ExceptionType::SpeculativeConstants));
        B(dispatcher_no_check);
        SwitchToNearCode();
      }

      {
        auto tmp = gpr.GetScopedReg();
        ARM64Reg value = gpr.R(i);
        MOVI2R(tmp, compile_time_value);
        CMP(value, tmp);
      }

      FixupBranch no_fail = B(CCFlags::CC_EQ);
      B(fail);
      SetJumpTarget(no_fail);

      // STV_GC_JITGF_v1: si el GPR es fijo, el CMP de arriba ya probo que su registro tiene el valor: dirty = false
      // evita un MOVI2R inutil al vaciarlo (en modo 2 la memoria tambien lo tiene: es igual al registro al entrar).
      gpr.SetImmediate(i, compile_time_value, !gpr.STVEsFijo(i));
    }
  }
}

void JitArm64::EmitUpdateMembase()
{
  LDR(IndexType::Unsigned, MEM_REG, PPC_REG, PPCSTATE_OFF(mem_ptr));
}

void JitArm64::MSRUpdated(u32 msr)
{
  // Update mem_ptr
  auto& memory = m_system.GetMemory();
  MOVP2R(MEM_REG,
         UReg_MSR(msr).DR ?
             (jo.fastmem ? memory.GetLogicalBase() : memory.GetLogicalPageMappingsBase()) :
             (jo.fastmem ? memory.GetPhysicalBase() : memory.GetPhysicalPageMappingsBase()));
  STR(IndexType::Unsigned, MEM_REG, PPC_REG, PPCSTATE_OFF(mem_ptr));

  // Update feature_flags
  static_assert(UReg_MSR{}.DR.StartBit() == 4);
  static_assert(UReg_MSR{}.IR.StartBit() == 5);
  static_assert(FEATURE_FLAG_MSR_DR == 1 << 0);
  static_assert(FEATURE_FLAG_MSR_IR == 1 << 1);
  const u32 other_feature_flags = m_ppc_state.feature_flags & ~(0x3 | STV_FEATURE_FLAG_MSR_FP);
  const u32 feature_flags = other_feature_flags | ((msr >> 4) & 0x3) |
                            ((g_stv_ff_fp && UReg_MSR(msr).FP) ? STV_FEATURE_FLAG_MSR_FP : 0);
  if (feature_flags == 0)
  {
    STR(IndexType::Unsigned, ARM64Reg::WZR, PPC_REG, PPCSTATE_OFF(feature_flags));
  }
  else
  {
    auto WA = gpr.GetScopedReg();
    MOVI2R(WA, feature_flags);
    STR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(feature_flags));
  }

  // Call PageTableUpdatedFromJit if needed
  if (UReg_MSR(msr).DR)
  {
    gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
    fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);

    auto WA = gpr.GetScopedReg();

    static_assert(PPCSTATE_OFF(pagetable_update_pending) < 0x1000);
    LDRB(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(pagetable_update_pending));
    FixupBranch update_not_pending = CBZ(WA);
    ABI_CallFunction(&PowerPC::MMU::PageTableUpdatedFromJit, &m_system.GetMMU());
    SetJumpTarget(update_not_pending);
  }
}

void JitArm64::MSRUpdated(ARM64Reg msr)
{
  auto WA = gpr.GetScopedReg();
  ARM64Reg XA = EncodeRegTo64(WA);

  // Update mem_ptr
  auto& memory = m_system.GetMemory();
  MOVP2R(MEM_REG, jo.fastmem ? memory.GetLogicalBase() : memory.GetLogicalPageMappingsBase());
  MOVP2R(XA, jo.fastmem ? memory.GetPhysicalBase() : memory.GetPhysicalPageMappingsBase());
  TST(msr, LogicalImm(1ULL << UReg_MSR{}.DR.StartBit(), GPRSize::B32));
  CSEL(MEM_REG, MEM_REG, XA, CCFlags::CC_NEQ);
  STR(IndexType::Unsigned, MEM_REG, PPC_REG, PPCSTATE_OFF(mem_ptr));

  // Update feature_flags
  static_assert(UReg_MSR{}.DR.StartBit() == 4);
  static_assert(UReg_MSR{}.IR.StartBit() == 5);
  static_assert(FEATURE_FLAG_MSR_DR == 1 << 0);
  static_assert(FEATURE_FLAG_MSR_IR == 1 << 1);
  const u32 other_feature_flags = m_ppc_state.feature_flags & ~(0x3 | STV_FEATURE_FLAG_MSR_FP);
  UBFX(WA, msr, 4, 2);
  if (other_feature_flags != 0)
    ORR(WA, WA, LogicalImm(other_feature_flags, GPRSize::B32));
  if (g_stv_ff_fp)  // STV_GC_FFFP_x: MSR.FP (bit 13) -> bandera bit 3
  {
    auto WF = gpr.GetScopedReg();
    UBFX(WF, msr, 13, 1);
    ORR(WA, WA, WF, ArithOption(WF, ShiftType::LSL, 3));
  }
  STR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(feature_flags));

  // Call PageTableUpdatedFromJit if needed
  MOV(WA, msr);
  gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
  fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
  FixupBranch dr_unset = TBZ(WA, u8(UReg_MSR{}.DR.StartBit()));
  static_assert(PPCSTATE_OFF(pagetable_update_pending) < 0x1000);
  LDRB(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(pagetable_update_pending));
  FixupBranch update_not_pending = CBZ(WA);
  ABI_CallFunction(&PowerPC::MMU::PageTableUpdatedFromJit, &m_system.GetMMU());
  SetJumpTarget(update_not_pending);
  SetJumpTarget(dr_unset);
}

void JitArm64::WriteExit(u32 destination, bool LK, u32 exit_address_after_return,
                         ARM64Reg exit_address_after_return_reg)
{
  Cleanup();
  if (IsProfilingEnabled())
  {
    ABI_CallFunction(&JitBlock::ProfileData::EndProfiling, js.curBlock->profile_data.get(),
                     js.downcountAmount);
  }
  DoDownCount();

  LK &= m_enable_blr_optimization;

  // STV_GC_CORTO_x
  const bool corto = STVCorto(destination);
  const bool corto_ret = LK && STVCorto(exit_address_after_return);
  const size_t tam = JitArm64BlockCache::STVTamEnlace(corto);
  const size_t tam_ret = JitArm64BlockCache::STVTamEnlace(corto_ret);

  const u8* host_address_after_return = nullptr;
  if (LK)
  {
    // Push {ARM_PC (64-bit); PPC_PC (32-bit); feature_flags (32-bit)} on the stack
    ARM64Reg reg_to_push = ARM64Reg::X1;
    const u64 feature_flags = m_ppc_state.feature_flags;
    if (exit_address_after_return_reg == ARM64Reg::INVALID_REG)
    {
      MOVI2R(ARM64Reg::X1, feature_flags << 32 | exit_address_after_return);
      // STV_GC_IC_x: los 32 bits bajos son el LR de bcctrl (salida enlazada al destino adivinado).
      if (m_stv_lr_en_salida)
        STR(IndexType::Unsigned, ARM64Reg::W1, PPC_REG, PPCSTATE_OFF_SPR(SPR_LR));
    }
    else if (feature_flags == 0)
    {
      reg_to_push = EncodeRegTo64(exit_address_after_return_reg);
    }
    else
    {
      ORRI2R(ARM64Reg::X1, EncodeRegTo64(exit_address_after_return_reg), feature_flags << 32,
             ARM64Reg::X1);
    }
    const s32 adr_offset = static_cast<s32>(tam + sizeof(u32) * 2);
    host_address_after_return = GetCodePtr() + adr_offset;
    ADR(ARM64Reg::X0, adr_offset);
    STP(IndexType::Pre, ARM64Reg::X0, reg_to_push, ARM64Reg::SP, -16);
  }

  constexpr size_t primary_farcode_size = 3 * sizeof(u32);
  const bool switch_to_far_code = !IsInFarCode();
  const u8* primary_farcode_addr;
  if (switch_to_far_code)
  {
    SwitchToFarCode();
    primary_farcode_addr = GetCodePtr();
    SwitchToNearCode();
  }
  else
  {
    primary_farcode_addr = GetCodePtr() + tam + (LK ? tam_ret : 0);
  }
  const u8* return_farcode_addr = primary_farcode_addr + primary_farcode_size;

  JitBlock* b = js.curBlock;
  JitBlock::LinkData linkData;
  linkData.exitAddress = destination;
  linkData.exitPtrs = GetWritableCodePtr();
  linkData.linkStatus = false;
  linkData.call = LK;
  linkData.exitFarcode = primary_farcode_addr;
  linkData.stv_corto = corto;
  b->linkData.push_back(linkData);

  blocks.WriteLinkBlock(*this, linkData);

  if (LK)
  {
    DEBUG_ASSERT(GetCodePtr() == host_address_after_return || HasWriteFailed());

    // Write the regular exit node after the return.
    linkData.exitAddress = exit_address_after_return;
    linkData.exitPtrs = GetWritableCodePtr();
    linkData.linkStatus = false;
    linkData.call = false;
    linkData.exitFarcode = return_farcode_addr;
    linkData.stv_corto = corto_ret;
    b->linkData.push_back(linkData);

    blocks.WriteLinkBlock(*this, linkData);
  }

  if (switch_to_far_code)
    SwitchToFarCode();
  DEBUG_ASSERT(GetCodePtr() == primary_farcode_addr || HasWriteFailed());
  MOVI2R(DISPATCHER_PC, destination);
  // STV_GC_CORTO_x: una salida corta llega aca sin enlazar (no por vencimiento): el despachador chequea las banderas.
  if (corto)
    B(dispatcher);
  else if (LK)
    BL(GetAsmRoutines()->do_timing);
  else
    B(GetAsmRoutines()->do_timing);

  if (LK)
  {
    if (GetCodePtr() == return_farcode_addr - sizeof(u32))
      BRK(101);
    DEBUG_ASSERT(GetCodePtr() == return_farcode_addr || HasWriteFailed());
    MOVI2R(DISPATCHER_PC, exit_address_after_return);
    B(corto_ret ? dispatcher : GetAsmRoutines()->do_timing);
  }

  if (switch_to_far_code)
    SwitchToNearCode();
}

void JitArm64::WriteExit(Arm64Gen::ARM64Reg dest, bool LK, u32 exit_address_after_return,
                         ARM64Reg exit_address_after_return_reg)
{
  // STV_GC_BLRFIX_v1: el MOV de abajo pisa DISPATCHER_PC (W26). Desde que STV_GC_JITDC_v1 saco W27 del pool, el
  // temporal con la direccion de retorno de bcctrl (WB, el primero del pool) es W26: se empujaba el DESTINO como
  // clave y TODO retorno de una llamada indirecta fallaba, vaciando la pila de la optimizacion BLR (Wind Waker:
  // 4,1 M de 8,4 M retornos/s; con el arreglo, 57 mil). En la 2609 oficial WB es W27 y no pasa. La direccion de
  // retorno se conoce al compilar: se empuja como constante.
  if (m_stv_blr_fix && exit_address_after_return_reg == DISPATCHER_PC && dest != DISPATCHER_PC)
    exit_address_after_return_reg = ARM64Reg::INVALID_REG;

  if (dest != DISPATCHER_PC)
    MOV(DISPATCHER_PC, dest);

  Cleanup();
  if (IsProfilingEnabled())
  {
    ABI_CallFunction(&JitBlock::ProfileData::EndProfiling, js.curBlock->profile_data.get(),
                     js.downcountAmount);
  }
  DoDownCount();

  LK &= m_enable_blr_optimization;

  if (!LK)
  {
    B(dispatcher);
  }
  else
  {
    // Push {ARM_PC (64-bit); PPC_PC (32-bit); feature_flags (32-bit)} on the stack
    ARM64Reg reg_to_push = ARM64Reg::X1;
    const u64 feature_flags = m_ppc_state.feature_flags;
    if (exit_address_after_return_reg == ARM64Reg::INVALID_REG)
    {
      MOVI2R(ARM64Reg::X1, feature_flags << 32 | exit_address_after_return);
      // STV_GC_LLAMADA_x: los 32 bits bajos son el LR de bcctrl.
      if (m_stv_lr_en_salida)
        STR(IndexType::Unsigned, ARM64Reg::W1, PPC_REG, PPCSTATE_OFF_SPR(SPR_LR));
    }
    else if (feature_flags == 0)
    {
      reg_to_push = EncodeRegTo64(exit_address_after_return_reg);
    }
    else
    {
      ORRI2R(ARM64Reg::X1, EncodeRegTo64(exit_address_after_return_reg), feature_flags << 32,
             ARM64Reg::X1);
    }
    constexpr s32 adr_offset = sizeof(u32) * 3;
    const u8* host_address_after_return = GetCodePtr() + adr_offset;
    ADR(ARM64Reg::X0, adr_offset);
    STP(IndexType::Pre, ARM64Reg::X0, reg_to_push, ARM64Reg::SP, -16);

    BL(dispatcher);
    DEBUG_ASSERT(GetCodePtr() == host_address_after_return || HasWriteFailed());

    // Write the regular exit node after the return.
    JitBlock* b = js.curBlock;
    JitBlock::LinkData linkData;
    linkData.exitAddress = exit_address_after_return;
    linkData.exitPtrs = GetWritableCodePtr();
    linkData.linkStatus = false;
    linkData.call = false;
    linkData.stv_corto = STVCorto(exit_address_after_return);  // STV_GC_CORTO_x
    const bool switch_to_far_code = !IsInFarCode();
    if (switch_to_far_code)
    {
      SwitchToFarCode();
      linkData.exitFarcode = GetCodePtr();
      SwitchToNearCode();
    }
    else
    {
      linkData.exitFarcode = GetCodePtr() + JitArm64BlockCache::STVTamEnlace(linkData.stv_corto);
    }
    b->linkData.push_back(linkData);

    blocks.WriteLinkBlock(*this, linkData);

    if (switch_to_far_code)
      SwitchToFarCode();
    MOVI2R(DISPATCHER_PC, exit_address_after_return);
    B(linkData.stv_corto ? dispatcher : GetAsmRoutines()->do_timing);
    if (switch_to_far_code)
      SwitchToNearCode();
  }
}

void JitArm64::FakeLKExit(u32 exit_address_after_return, ARM64Reg exit_address_after_return_reg)
{
  if (!m_enable_blr_optimization)
    return;

  // We may need to fake the BLR stack on inlined CALL instructions.
  // Else we can't return to this location any more.
  if (exit_address_after_return_reg != ARM64Reg::W30)
  {
    // Do not lock W30 if it is the same as the exit address register, since
    // it's already locked. It'll only get clobbered at the BL (below) where
    // we do not need its value anymore.
    // NOTE: This means W30 won't contain the return address anymore after this
    // function has been called!
    gpr.Lock(ARM64Reg::W30);
  }

  const u8* host_address_after_return;
  {
    // Push {ARM_PC (64-bit); PPC_PC (32-bit); feature_flags (32-bit)} on the stack
    Arm64RegCache::ScopedARM64Reg after_reg;
    ARM64Reg reg_to_push;
    const u64 feature_flags = m_ppc_state.feature_flags;
    if (exit_address_after_return_reg == ARM64Reg::INVALID_REG)
    {
      after_reg = gpr.GetScopedReg();
      reg_to_push = EncodeRegTo64(after_reg);
      MOVI2R(reg_to_push, feature_flags << 32 | exit_address_after_return);
    }
    else if (feature_flags == 0)
    {
      reg_to_push = EncodeRegTo64(exit_address_after_return_reg);
    }
    else
    {
      after_reg = gpr.GetScopedReg();
      reg_to_push = EncodeRegTo64(after_reg);
      ORRI2R(reg_to_push, EncodeRegTo64(exit_address_after_return_reg), feature_flags << 32,
             reg_to_push);
    }

    auto code_reg = gpr.GetScopedReg();
    constexpr s32 adr_offset = sizeof(u32) * 3;
    host_address_after_return = GetCodePtr() + adr_offset;
    ADR(EncodeRegTo64(code_reg), adr_offset);
    STP(IndexType::Pre, EncodeRegTo64(code_reg), reg_to_push, ARM64Reg::SP, -16);
  }

  FixupBranch skip_exit = BL();
  DEBUG_ASSERT(GetCodePtr() == host_address_after_return || HasWriteFailed());
  if (exit_address_after_return_reg != ARM64Reg::W30)
  {
    gpr.Unlock(ARM64Reg::W30);
  }

  // Write the regular exit node after the return.
  JitBlock* b = js.curBlock;
  JitBlock::LinkData linkData;
  linkData.exitAddress = exit_address_after_return;
  linkData.exitPtrs = GetWritableCodePtr();
  linkData.linkStatus = false;
  linkData.call = false;
  linkData.stv_corto = STVCorto(exit_address_after_return);  // STV_GC_CORTO_x
  const bool switch_to_far_code = !IsInFarCode();
  if (switch_to_far_code)
  {
    SwitchToFarCode();
    linkData.exitFarcode = GetCodePtr();
    SwitchToNearCode();
  }
  else
  {
    linkData.exitFarcode = GetCodePtr() + JitArm64BlockCache::STVTamEnlace(linkData.stv_corto);
  }
  b->linkData.push_back(linkData);

  blocks.WriteLinkBlock(*this, linkData);

  if (switch_to_far_code)
    SwitchToFarCode();
  MOVI2R(DISPATCHER_PC, exit_address_after_return);
  B(linkData.stv_corto ? dispatcher : GetAsmRoutines()->do_timing);
  if (switch_to_far_code)
    SwitchToNearCode();

  SetJumpTarget(skip_exit);
}

void JitArm64::WriteBLRExit(Arm64Gen::ARM64Reg dest)
{
  if (!m_enable_blr_optimization)
  {
    WriteExit(dest);
    return;
  }

  if (dest != DISPATCHER_PC)
    MOV(DISPATCHER_PC, dest);

  Cleanup();
  if (IsProfilingEnabled())
  {
    ABI_CallFunction(&JitBlock::ProfileData::EndProfiling, js.curBlock->profile_data.get(),
                     js.downcountAmount);
  }

  // Check if {PPC_PC, feature_flags} matches the current state, then RET to ARM_PC.
  LDP(IndexType::Post, ARM64Reg::X2, ARM64Reg::X1, ARM64Reg::SP, 16);
  const u64 feature_flags = m_ppc_state.feature_flags;
  // STV_GC_BLRCORTO_x (bit 32 de la valvula): la comparacion va por EOR + CBNZ (no toca las banderas), asi el SUBS del
  // downcount es uno solo y el camino de fallo (ResetStack + despachador) pasa al codigo lejano. X3 esta libre (todo
  // vaciado) y DoDownCount puede pisar X0/X1.
  if (m_stv_volatil & 32)
  {
    if (feature_flags == 0)
    {
      EOR(ARM64Reg::X3, ARM64Reg::X1, EncodeRegTo64(DISPATCHER_PC));
    }
    else
    {
      ORRI2R(ARM64Reg::X0, EncodeRegTo64(DISPATCHER_PC), feature_flags << 32, ARM64Reg::X0);
      EOR(ARM64Reg::X3, ARM64Reg::X1, ARM64Reg::X0);
    }
    DoDownCount();
    FixupBranch fallo = CBNZ(ARM64Reg::X3);
    RET(ARM64Reg::X2);
    SetJumpTarget(fallo);
    FixupBranch lejos = B();
    SwitchToFarCode();
    SetJumpTarget(lejos);
    ResetStack();
    if (m_stv_blr_sin_and)
      AND(DISPATCHER_PC, DISPATCHER_PC, LogicalImm(~0x3, GPRSize::B32));
    B(dispatcher);
    SwitchToNearCode();
    return;
  }
  if (feature_flags == 0)
  {
    CMP(ARM64Reg::X1, EncodeRegTo64(DISPATCHER_PC));
  }
  else
  {
    ORRI2R(ARM64Reg::X0, EncodeRegTo64(DISPATCHER_PC), feature_flags << 32, ARM64Reg::X0);
    CMP(ARM64Reg::X1, ARM64Reg::X0);
  }
  FixupBranch no_match = B(CC_NEQ);

  DoDownCount();  // overwrites X0 + X1

  RET(ARM64Reg::X2);

  SetJumpTarget(no_match);

  ResetStack();

  DoDownCount();

  // STV_GC_LLAMADA_x: el destino llego sin enmascarar (ver bclrx).
  if (m_stv_blr_sin_and)
    AND(DISPATCHER_PC, DISPATCHER_PC, LogicalImm(~0x3, GPRSize::B32));

  B(dispatcher);
}

void JitArm64::WriteExceptionExit(u32 destination, bool only_external, bool always_exception)
{
  MOVI2R(DISPATCHER_PC, destination);
  WriteExceptionExit(DISPATCHER_PC, only_external, always_exception);
}

void JitArm64::WriteExceptionExit(ARM64Reg dest, bool only_external, bool always_exception)
{
  if (dest != DISPATCHER_PC)
    MOV(DISPATCHER_PC, dest);

  Cleanup();

  FixupBranch no_exceptions;
  if (!always_exception)
  {
    LDR(IndexType::Unsigned, ARM64Reg::W30, PPC_REG, PPCSTATE_OFF(Exceptions));
    no_exceptions = CBZ(ARM64Reg::W30);
  }

  static_assert(PPCSTATE_OFF(pc) <= 252);
  static_assert(PPCSTATE_OFF(pc) + 4 == PPCSTATE_OFF(npc));
  STP(IndexType::Signed, DISPATCHER_PC, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(pc));

  const auto f =
      only_external ? &PowerPC::CheckExternalExceptionsFromJIT : &PowerPC::CheckExceptionsFromJIT;
  ABI_CallFunction(f, &m_system.GetPowerPC());

  EmitUpdateMembase();

  LDR(IndexType::Unsigned, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(npc));

  if (!always_exception)
    SetJumpTarget(no_exceptions);

  if (IsProfilingEnabled())
  {
    ABI_CallFunction(&JitBlock::ProfileData::EndProfiling, js.curBlock->profile_data.get(),
                     js.downcountAmount);
  }
  DoDownCount();

  B(dispatcher);
}

void JitArm64::WriteConditionalExceptionExit(int exception, u64 increment_sp_on_exit)
{
  auto WA = gpr.GetScopedReg();
  WriteConditionalExceptionExit(exception, WA, Arm64Gen::ARM64Reg::INVALID_REG,
                                increment_sp_on_exit);
}

void JitArm64::WriteConditionalExceptionExit(int exception, ARM64Reg temp_gpr, ARM64Reg temp_fpr,
                                             u64 increment_sp_on_exit)
{
  LDR(IndexType::Unsigned, temp_gpr, PPC_REG, PPCSTATE_OFF(Exceptions));
  FixupBranch no_exception = TBZ(temp_gpr, MathUtil::IntLog2(exception));

  const bool switch_to_far_code = !IsInFarCode();

  if (switch_to_far_code)
  {
    FixupBranch handle_exception = B();
    SwitchToFarCode();
    SetJumpTarget(handle_exception);
  }

  if (increment_sp_on_exit != 0)
    ADDI2R(ARM64Reg::SP, ARM64Reg::SP, increment_sp_on_exit, temp_gpr);

  // STV_GC_VOLATIL_x: un volatil muerto puede estar descartado en una salida por excepcion.
  const IgnoreDiscardedRegisters stv_ign =
      (m_stv_volatil & 2) ? IgnoreDiscardedRegisters::Yes : IgnoreDiscardedRegisters::No;
  gpr.Flush(FlushMode::MaintainState, temp_gpr, stv_ign);
  fpr.Flush(FlushMode::MaintainState, temp_fpr, stv_ign);

  WriteExceptionExit(js.compilerPC, false, true);

  if (switch_to_far_code)
    SwitchToNearCode();

  SetJumpTarget(no_exception);
}

bool JitArm64::HandleFunctionHooking(u32 address)
{
  const auto result = HLE::TryReplaceFunction(m_ppc_symbol_db, address, PowerPC::CoreMode::JIT);
  if (!result)
    return false;
  if (!HLE::STVHookValido(m_system, result.hook_index))  // STV_GC_SKINHLE_v1
    return false;

  HLEFunction(result.hook_index);

  if (result.type != HLE::HookType::Replace)
    return false;

  LDR(IndexType::Unsigned, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(npc));
  js.downcountAmount += js.st.numCycles;
  WriteExit(DISPATCHER_PC);
  return true;
}

void JitArm64::DumpCode(const u8* start, const u8* end)
{
  std::string output;
  for (const u8* code = start; code < end; code += sizeof(u32))
    output += fmt::format("{:08x}", Common::swap32(code));
  WARN_LOG_FMT(DYNA_REC, "Code dump from {} to {}:\n{}", fmt::ptr(start), fmt::ptr(end), output);
}

void JitArm64::Run()
{
  ProtectStack();
  m_system.GetJitInterface().UpdateMembase();

  CompiledCode pExecAddr = (CompiledCode)enter_code;
  pExecAddr();

  UnprotectStack();
}

void JitArm64::SingleStep()
{
  ProtectStack();
  m_system.GetJitInterface().UpdateMembase();

  CompiledCode pExecAddr = (CompiledCode)enter_code;
  pExecAddr();

  UnprotectStack();
}

void JitArm64::Trace()
{
  std::string regs;
  std::string fregs;

#ifdef JIT_LOG_GPR
  for (size_t i = 0; i < std::size(m_ppc_state.gpr); i++)
  {
    regs += fmt::format("r{:02d}: {:08x} ", i, m_ppc_state.gpr[i]);
  }
#endif

#ifdef JIT_LOG_FPR
  for (size_t i = 0; i < std::size(m_ppc_state.ps); i++)
  {
    fregs += fmt::format("f{:02d}: {:016x} ", i, m_ppc_state.ps[i].PS0AsU64());
  }
#endif

  DEBUG_LOG_FMT(DYNA_REC,
                "JitArm64 PC: {:08x} SRR0: {:08x} SRR1: {:08x} FPSCR: {:08x} "
                "MSR: {:08x} LR: {:08x} {} {}",
                m_ppc_state.pc, SRR0(m_ppc_state), SRR1(m_ppc_state), m_ppc_state.fpscr.Hex,
                m_ppc_state.msr.Hex, m_ppc_state.spr[8], regs, fregs);
}

void JitArm64::Jit(u32 em_address)
{
  Jit(em_address, true);
}

void JitArm64::Jit(u32 em_address, bool clear_cache_and_retry_on_failure)
{
  CleanUpAfterStackFault();

  if (SConfig::GetInstance().bJITNoBlockCache)
    ClearCache();
  FreeRanges();

  const Common::ScopedJITPageWriteAndNoExecute enable_jit_page_writes;

  std::size_t block_size = m_code_buffer.size();

  auto& cpu = m_system.GetCPU();

  if (IsDebuggingEnabled())
  {
    // We can link blocks as long as we are not single stepping
    SetBlockLinkingEnabled(true);
    SetOptimizationEnabled(true);

    if (!IsProfilingEnabled())
    {
      if (cpu.IsStepping())
      {
        block_size = 1;

        // Do not link this block to other blocks while single stepping
        SetBlockLinkingEnabled(false);
        SetOptimizationEnabled(false);
      }
      Trace();
    }
  }

  // Analyze the block, collect all instructions it is made of (including inlining,
  // if that is enabled), reorder instructions for optimal performance, and join joinable
  // instructions.
  u32 nextPC = analyzer.Analyze(em_address, &code_block, &m_code_buffer, block_size);

  // STV_GC_GECKOJIT_v1: las opciones STV del analizador suponen codigo de compilador con su ABI (volatiles muertos en
  // el blr final, saltos internos hacia adelante). El gestor de trucos de Dolphin y los codigos C2 (zona
  // INSTALLER_BASE_ADDRESS..INSTALLER_END_ADDRESS) son ensamblador a mano que guarda y devuelve los volatiles: con esas
  // opciones el JIT descartaba sus restauraciones ("Attempted to flush discarded register" con cualquier truco). Un
  // bloque que toca esa zona (tambien por seguir ramas desde el juego) se analiza otra vez sin las opciones STV.
  constexpr u32 stv_opciones = PPCAnalyst::PPCAnalyzer::OPTION_STV_LSU_SIN_EXCEPCION |
                               PPCAnalyst::PPCAnalyzer::OPTION_STV_SALTO_INTERNO |
                               PPCAnalyst::PPCAnalyzer::OPTION_STV_CR_VOLATIL |
                               PPCAnalyst::PPCAnalyzer::OPTION_STV_HOJA |
                               PPCAnalyst::PPCAnalyzer::OPTION_STV_VOLATILES_MUERTOS;
  if (analyzer.STVOpciones() & stv_opciones)
  {
    bool stv_gecko = false;
    for (u32 i = 0; i < code_block.m_num_instructions && !stv_gecko; i++)
    {
      const u32 a = m_code_buffer[i].address;
      stv_gecko = a >= Gecko::INSTALLER_BASE_ADDRESS && a < Gecko::INSTALLER_END_ADDRESS;
    }
    if (stv_gecko)
    {
      const u32 antes = analyzer.STVOpciones();
      analyzer.STVPonerOpciones(antes & ~stv_opciones);
      nextPC = analyzer.Analyze(em_address, &code_block, &m_code_buffer, block_size);
      analyzer.STVPonerOpciones(antes);
#ifdef __ANDROID__
      static bool stv_avisado = false;
      if (!stv_avisado)
      {
        stv_avisado = true;
        __android_log_print(ANDROID_LOG_INFO, "STVGC", "jit: geckojit bloque %08x sin opciones STV", em_address);
      }
#endif
    }
  }

  if (code_block.m_memory_exception)
  {
    // Address of instruction could not be translated
    m_ppc_state.npc = nextPC;
    m_ppc_state.Exceptions |= EXCEPTION_ISI;
    m_system.GetPowerPC().CheckExceptions();
    m_system.GetJitInterface().UpdateMembase();
    WARN_LOG_FMT(POWERPC, "ISI exception at {:#010x}", nextPC);
    return;
  }

  if (std::optional<size_t> code_region_index = SetEmitterStateToFreeCodeRegion())
  {
    u8* near_start = GetWritableCodePtr();
    u8* far_start = m_far_code.GetWritableCodePtr();

    JitBlock* b = blocks.AllocateBlock(em_address);
    if (DoJit(em_address, b, nextPC))
    {
      // Code generation succeeded.

      // Mark the memory regions that this code block uses as used in the local rangesets.
      u8* near_end = GetWritableCodePtr();
      if (near_start != near_end)
      {
        (code_region_index == 0 ? m_free_ranges_near_0 : m_free_ranges_near_1)
            .erase(near_start, near_end);
      }
      u8* far_end = m_far_code.GetWritableCodePtr();
      if (far_start != far_end)
      {
        (code_region_index == 0 ? m_free_ranges_far_0 : m_free_ranges_far_1)
            .erase(far_start, far_end);
      }

      // Store the used memory regions in the block so we know what to mark as unused when the
      // block gets invalidated.
      b->near_begin = near_start;
      b->near_end = near_end;
      b->far_begin = far_start;
      b->far_end = far_end;

      blocks.FinalizeBlock(*b, jo.enableBlocklink, code_block, m_code_buffer);

#ifdef JIT_LOG_GENERATED_CODE
      LogGeneratedCode();
#endif
      return;
    }
  }

  if (clear_cache_and_retry_on_failure)
  {
    // Code generation failed due to not enough free space in either the near or far code regions.
    // Clear the entire JIT cache and retry.
    WARN_LOG_FMT(DYNA_REC, "flushing code caches, please report if this happens a lot");
    ClearCache();
    Jit(em_address, false);
    return;
  }

  PanicAlertFmtT("JIT failed to find code space after a cache clear. This should never happen. "
                 "Please report this incident on the bug tracker. Dolphin will now exit.");
  exit(-1);
}

void JitArm64::EraseSingleBlock(const JitBlock& block)
{
  blocks.EraseSingleBlock(block);
  FreeRanges();
}

std::vector<JitBase::MemoryStats> JitArm64::GetMemoryStats() const
{
  return {{"near_0", m_free_ranges_near_0.get_stats()},
          {"near_1", m_free_ranges_near_1.get_stats()},
          {"far_0", m_free_ranges_far_0.get_stats()},
          {"far_1", m_free_ranges_far_1.get_stats()}};
}

std::size_t JitArm64::DisassembleNearCode(const JitBlock& block, std::ostream& stream) const
{
  return m_disassembler->Disassemble(block.normalEntry, block.near_end, stream);
}

std::size_t JitArm64::DisassembleFarCode(const JitBlock& block, std::ostream& stream) const
{
  return m_disassembler->Disassemble(block.far_begin, block.far_end, stream);
}

std::optional<size_t> JitArm64::SetEmitterStateToFreeCodeRegion()
{
  // Find some large free memory blocks and set code emitters to point at them. If we can't find
  // free blocks, return std::nullopt instead, which will trigger a JIT cache clear.
  const auto free_near_0 = m_free_ranges_near_0.by_size_begin();
  const auto free_near_1 = m_free_ranges_near_1.by_size_begin();
  const auto free_far_0 = m_free_ranges_far_0.by_size_begin();
  const auto free_far_1 = m_free_ranges_far_1.by_size_begin();

  const size_t free_near_1_size = free_near_1.to() - free_near_1.from();
  const size_t free_far_1_size = free_far_1.to() - free_far_1.from();
  const size_t free_1_smallest_size = std::min(free_near_1_size, free_far_1_size);

  if (free_1_smallest_size >= 1024 * 1024)
  {
    // Don't use region 0 unless region 1 is getting full. This improves cache friendliness.
    SetCodePtr(free_near_1.from(), free_near_1.to());
    m_far_code.SetCodePtr(free_far_1.from(), free_far_1.to());
    return std::make_optional(1);
  }

  const size_t free_near_0_size = free_near_0.to() - free_near_0.from();
  const size_t free_far_0_size = free_far_0.to() - free_far_0.from();
  const size_t free_0_smallest_size = std::min(free_near_0_size, free_far_0_size);

  if (free_0_smallest_size == 0 && free_1_smallest_size == 0)
  {
    if (free_near_0_size == 0 && free_near_1_size == 0)
      WARN_LOG_FMT(DYNA_REC, "Failed to find free memory region in near code regions.");
    else if (free_far_0_size == 0 && free_far_1_size == 0)
      WARN_LOG_FMT(DYNA_REC, "Failed to find free memory region in far code regions.");
    return std::nullopt;
  }

  if (free_0_smallest_size > free_1_smallest_size)
  {
    SetCodePtr(free_near_0.from(), free_near_0.to());
    m_far_code.SetCodePtr(free_far_0.from(), free_far_0.to());
    return std::make_optional(0);
  }
  else
  {
    SetCodePtr(free_near_1.from(), free_near_1.to());
    m_far_code.SetCodePtr(free_far_1.from(), free_far_1.to());
    return std::make_optional(1);
  }
}

// STV_GC_AUDITA_x (instrumento): bytes de codigo cercano y lejano que emite cada instruccion PowerPC (con sus chequeos
// previos y vaciados posteriores), por bloque. "<entrada>" es lo emitido antes de la primera instruccion y "<fin>" lo
// emitido despues de la ultima. Lo prende y vuelca debug.stv.gc.audita (PerformanceMetrics); solo el hilo de CPU.
bool g_stv_audita = false;
std::unordered_map<u32, std::vector<STVAuditaOp>> g_stv_audita_bloques;

bool JitArm64::DoJit(u32 em_address, JitBlock* b, u32 nextPC)
{
  auto& cpu = m_system.GetCPU();

  js.isLastInstruction = false;
  js.firstFPInstructionFound = false;
  js.assumeNoPairedQuantize = false;
  js.constantGqrValid = BitSet8();  // STV_GC_GQR_v1
  js.blockStart = em_address;
  js.fifoBytesSinceCheck = 0;
  js.mustCheckFifo = false;
  js.downcountAmount = 0;
  js.skipInstructions = 0;
  js.curBlock = b;
  js.carryFlag = CarryFlag::InPPCState;
  js.numLoadStoreInst = 0;
  js.numFloatingPointInst = 0;
  m_stv_saltos.clear();  // STV_GC_SALTOINT_x
  m_stv_base_guest = {-1, -1};  // STV_GC_BASE_x
  m_stv_talones.clear();  // STV_GC_TALON_x

  b->normalEntry = GetWritableCodePtr();
  // STV_GC_AUDITA_x
  std::vector<STVAuditaOp>* stv_aud = nullptr;
  const u8* stv_n0 = GetCodePtr();
  const u8* stv_f0 = m_far_code.GetCodePtr();
  if (g_stv_audita)
  {
    stv_aud = &g_stv_audita_bloques[em_address];
    stv_aud->clear();
  }

  // STV_GC_JITGF_v1: modo 2, verificacion de los fijos contra ppcState al entrar.
  if (STVGf() == 2)
    STVGfVerificar();

  // Conditionally add profiling code.
  if (IsProfilingEnabled())
    ABI_CallFunction(&JitBlock::ProfileData::BeginProfiling, b->profile_data.get());

  if (code_block.m_gqr_used.Count() == 1 && !js.pairedQuantizeAddresses.contains(js.blockStart))
  {
    int gqr = *code_block.m_gqr_used.begin();
    if (!code_block.m_gqr_modified[gqr] && !GQR(m_ppc_state, gqr))
    {
      LDR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF_SPR(SPR_GQR0 + gqr));
      FixupBranch no_fail = CBZ(ARM64Reg::W0);
      FixupBranch fail = B();
      SwitchToFarCode();
      SetJumpTarget(fail);
      MOVI2R(DISPATCHER_PC, js.blockStart);
      STR(IndexType::Unsigned, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(pc));
      ABI_CallFunction(&JitInterface::CompileExceptionCheckFromJIT, &m_system.GetJitInterface(),
                       static_cast<u32>(JitInterface::ExceptionType::PairedQuantize));
      B(dispatcher_no_check);
      SwitchToNearCode();
      SetJumpTarget(no_fail);
      js.assumeNoPairedQuantize = true;
    }
  }

  // STV_GC_GQR_v1: los GQR que el bloque usa y no modifica se toman como constantes (el valor de ahora) con una
  // guardia al entrar, como Jit64 (ComputeStaticGQRs): psq_l los decuantiza en linea en vez de llamar a la rutina por
  // tabla. Si un GQR cambio, el bloque se anota en pairedQuantizeAddresses y se recompila sin suponer nada.
  if (m_stv_gqr && !js.assumeNoPairedQuantize && !jo.memcheck && jo.fastmem && code_block.m_gqr_used &&
      !js.pairedQuantizeAddresses.contains(js.blockStart))
  {
    BitSet8 fijos;
    for (int gqr : code_block.m_gqr_used)
    {
      if (!code_block.m_gqr_modified[gqr])
        fijos[gqr] = true;
    }
    if (fijos)
    {
      std::vector<FixupBranch> fallas;
      for (int gqr : fijos)
      {
        const u32 valor = GQR(m_ppc_state, gqr);
        js.constantGqr[gqr] = valor;
        LDR(IndexType::Unsigned, ARM64Reg::W0, PPC_REG, PPCSTATE_OFF_SPR(SPR_GQR0 + gqr));
        MOVI2R(ARM64Reg::W1, valor);
        CMP(ARM64Reg::W0, ARM64Reg::W1);
        FixupBranch igual = B(CC_EQ);
        fallas.push_back(B());
        SetJumpTarget(igual);
      }
      SwitchToFarCode();
      for (FixupBranch& f : fallas)
        SetJumpTarget(f);
      MOVI2R(DISPATCHER_PC, js.blockStart);
      STR(IndexType::Unsigned, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(pc));
      ABI_CallFunction(&JitInterface::CompileExceptionCheckFromJIT, &m_system.GetJitInterface(),
                       static_cast<u32>(JitInterface::ExceptionType::PairedQuantize));
      B(dispatcher_no_check);
      SwitchToNearCode();
      js.constantGqrValid = fijos;
    }
  }

  gpr.Start(js.gpa);
  fpr.Start(js.fpa);

  m_constant_propagation.Clear();

  if (!js.noSpeculativeConstantsAddresses.contains(js.blockStart))
  {
    IntializeSpeculativeConstants();
  }

  // Translate instructions
  if (stv_aud)
  {
    stv_aud->push_back({"<entrada>", em_address, 0, static_cast<u32>(GetCodePtr() - stv_n0),
                        static_cast<u32>(m_far_code.GetCodePtr() - stv_f0)});
  }
  for (u32 i = 0; i < code_block.m_num_instructions; i++)
  {
    const u8* stv_in = GetCodePtr();
    const u8* stv_if = m_far_code.GetCodePtr();
    PPCAnalyst::CodeOp& op = m_code_buffer[i];

    // STV_GC_SALTOINT_x
    if (op.stvEtiqueta)
      STVEtiqueta(i);

    js.compilerPC = op.address;
    js.op = &op;
    js.fpr_is_store_safe = op.fprIsStoreSafeBeforeInst;
    js.instructionsLeft = (code_block.m_num_instructions - 1) - i;
    const GekkoOPInfo* opinfo = op.opinfo;
    js.downcountAmount += opinfo->num_cycles;
    js.isLastInstruction = i == (code_block.m_num_instructions - 1);

    // Skip calling UpdateLastUsed for lmw/stmw - it usually hurts more than it helps
    if (op.inst.OPCD != 46 && op.inst.OPCD != 47)
      gpr.UpdateLastUsed(op.regsIn | op.regsOut);

    BitSet32 fpr_used = op.fregsIn;
    if (op.fregOut >= 0)
      fpr_used[op.fregOut] = true;
    fpr.UpdateLastUsed(fpr_used);

    if (i != 0)
    {
      // Gather pipe writes using a non-immediate address are discovered by profiling.
      const u32 prev_address = m_code_buffer[i - 1].address;
      bool gatherPipeIntCheck = js.fifoWriteAddresses.contains(prev_address);

      // STV_GC_FIFORUT_v1 (tope de seguridad de FIFO3): al final de un bloque el gather pipe puede quedar con hasta
      // umbral-1 bytes (Cleanup solo vacia sobre el umbral), y este contador arranca en 0 en el bloque siguiente; mas
      // una escritura de hasta 8 bytes que no cuenta el compilador (camino lento o rutina) antes de vaciar. Con el
      // umbral de 256 eso pasaba los 512 del bufer (m_gather_pipe). El chequeo dentro del bloque se adelanta lo justo:
      // umbral-1 + (tope-1+8) + 8 < GATHER_PIPE_EXTRA_SIZE. Con 32 (Dolphin) no cambia nada.
      const u32 stv_tope = std::min<u32>(m_stv_gp_umbral, GPFifo::GATHER_PIPE_EXTRA_SIZE - m_stv_gp_umbral - 24);
      if (jo.optimizeGatherPipe &&
          (js.fifoBytesSinceCheck >= stv_tope || js.mustCheckFifo))  // STV_GC_FIFO3_v1
      {
        js.fifoBytesSinceCheck = 0;
        js.mustCheckFifo = false;

        gpr.Lock(ARM64Reg::W30);
        BitSet32 regs_in_use = gpr.GetCallerSavedUsed();
        BitSet32 fprs_in_use = fpr.GetCallerSavedUsed();
        regs_in_use[DecodeReg(ARM64Reg::W30)] = false;

        ABI_PushRegisters(regs_in_use);
        m_float_emit.ABI_PushRegisters(fprs_in_use, ARM64Reg::X30);
        ABI_CallFunction(&GPFifo::FastCheckGatherPipe, &m_system.GetGPFifo());
        m_float_emit.ABI_PopRegisters(fprs_in_use, ARM64Reg::X30);
        ABI_PopRegisters(regs_in_use);

        gpr.Unlock(ARM64Reg::W30);
        gatherPipeIntCheck = true;
      }
      // Gather pipe writes can generate an exception; add an exception check.
      // TODO: This doesn't really match hardware; the CP interrupt is
      // asynchronous.
      if (jo.optimizeGatherPipe && gatherPipeIntCheck)
      {
        auto WA = gpr.GetScopedReg();
        ARM64Reg XA = EncodeRegTo64(WA);

        LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(Exceptions));
        FixupBranch no_ext_exception = TBZ(WA, MathUtil::IntLog2(EXCEPTION_EXTERNAL_INT));
        FixupBranch exception = B();
        SwitchToFarCode();
        const u8* done_here = GetCodePtr();
        FixupBranch exit = B();
        SetJumpTarget(exception);
        LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(msr));
        TBZ(WA, 15, done_here);  // MSR.EE
        LDR(IndexType::Unsigned, WA, XA,
            MOVPage2R(XA, &m_system.GetProcessorInterface().m_interrupt_cause));
        constexpr u32 cause_mask = ProcessorInterface::INT_CAUSE_CP |
                                   ProcessorInterface::INT_CAUSE_PE_TOKEN |
                                   ProcessorInterface::INT_CAUSE_PE_FINISH;
        TST(WA, LogicalImm(cause_mask, GPRSize::B32));
        B(CC_EQ, done_here);

        gpr.Flush(FlushMode::MaintainState, WA,
                  (m_stv_volatil & 2) ? IgnoreDiscardedRegisters::Yes : IgnoreDiscardedRegisters::No);
        fpr.Flush(FlushMode::MaintainState, ARM64Reg::INVALID_REG,
                  (m_stv_volatil & 2) ? IgnoreDiscardedRegisters::Yes : IgnoreDiscardedRegisters::No);
        WriteExceptionExit(js.compilerPC, true, true);
        SwitchToNearCode();
        SetJumpTarget(no_ext_exception);
        SetJumpTarget(exit);
      }
    }

    if (HandleFunctionHooking(op.address))
      break;

    if (op.skip)
    {
      if (IsBranchWatchEnabled())
      {
        // The only thing that currently sets op.skip is the BLR following optimization.
        // If any non-branch instruction starts setting that too, this will need to be changed.
        ASSERT(op.inst.hex == 0x4e800020);
        WriteBranchWatch<true>(op.address, op.branchTo, op.inst, gpr.GetCallerSavedUsed(),
                               fpr.GetCallerSavedUsed());
      }
    }
    else
    {
      if (IsDebuggingEnabled() && !cpu.IsStepping() &&
          m_system.GetPowerPC().GetBreakPoints().IsAddressBreakPoint(op.address))
      {
        FlushCarry();
        gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
        fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);

        static_assert(PPCSTATE_OFF(pc) <= 252);
        static_assert(PPCSTATE_OFF(pc) + 4 == PPCSTATE_OFF(npc));

        MOVI2R(DISPATCHER_PC, op.address);
        STP(IndexType::Signed, DISPATCHER_PC, DISPATCHER_PC, PPC_REG, PPCSTATE_OFF(pc));
        // STV_GC_JITGF_v1: las condiciones de los puntos de parada leen gpr (y una asignacion puede escribirlo), y el
        // registro de log imprime r3-r12. Si para, el B(dispatcher_exit) de abajo vuelve a guardar (inocuo).
        STVGfGuardar();
        ABI_CallFunction(&PowerPC::CheckAndHandleBreakPointsFromJIT, &m_system.GetPowerPC());
        STVGfCargar();

        LDR(IndexType::Unsigned, ARM64Reg::W0, ARM64Reg::X0,
            MOVPage2R(ARM64Reg::X0, cpu.GetStatePtr()));
        static_assert(std::to_underlying(CPU::State::Running) == 0);
        FixupBranch no_breakpoint = CBZ(ARM64Reg::W0);

        Cleanup();
        if (IsProfilingEnabled())
        {
          ABI_CallFunction(&JitBlock::ProfileData::EndProfiling, b->profile_data.get(),
                           js.downcountAmount);
        }
        DoDownCount();
        B(dispatcher_exit);

        SetJumpTarget(no_breakpoint);
      }

      // STV_GC_FFFP_x: el bloque se compilo con MSR.FP = 1 (bandera de bloque): no hace falta el chequeo.
      if ((opinfo->flags & FL_USE_FPU) && !js.firstFPInstructionFound && g_stv_ff_fp &&
          (m_ppc_state.feature_flags & STV_FEATURE_FLAG_MSR_FP))
      {
        js.firstFPInstructionFound = true;
      }
      if ((opinfo->flags & FL_USE_FPU) && !js.firstFPInstructionFound)
      {
        FixupBranch b1;
        // This instruction uses FPU - needs to add FP exception bailout
        {
          auto WA = gpr.GetScopedReg();
          LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(msr));
          b1 = TBNZ(WA, 13);  // Test FP enabled bit

          FixupBranch far_addr = B();
          SwitchToFarCode();
          SetJumpTarget(far_addr);

          gpr.Flush(FlushMode::MaintainState, WA,
                    (m_stv_volatil & 2) ? IgnoreDiscardedRegisters::Yes : IgnoreDiscardedRegisters::No);
          fpr.Flush(FlushMode::MaintainState, ARM64Reg::INVALID_REG,
                    (m_stv_volatil & 2) ? IgnoreDiscardedRegisters::Yes : IgnoreDiscardedRegisters::No);

          LDR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(Exceptions));
          ORR(WA, WA, LogicalImm(EXCEPTION_FPU_UNAVAILABLE, GPRSize::B32));
          STR(IndexType::Unsigned, WA, PPC_REG, PPCSTATE_OFF(Exceptions));
        }

        WriteExceptionExit(js.compilerPC, false, true);

        SwitchToNearCode();

        SetJumpTarget(b1);

        js.firstFPInstructionFound = true;
      }

      if (bJITRegisterCacheOff)
      {
        FlushCarry();
        gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
        fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
        m_constant_propagation.Clear();

        CompileInstruction(op);
      }
      else
      {
        const JitCommon::ConstantPropagationResult constant_propagation_result =
            m_constant_propagation.EvaluateInstruction(op.inst, opinfo->flags);

        if (!constant_propagation_result.instruction_fully_executed)
          CompileInstruction(op);

        m_constant_propagation.Apply(constant_propagation_result);

        if (constant_propagation_result.gpr >= 0)
        {
          // Mark the GPR as dirty in the register cache
          gpr.SetImmediate(constant_propagation_result.gpr, constant_propagation_result.gpr_value);
        }

        if (constant_propagation_result.instruction_fully_executed)
        {
          if (constant_propagation_result.carry)
            ComputeCarry(*constant_propagation_result.carry);

          if (constant_propagation_result.overflow)
            GenerateConstantOverflow(*constant_propagation_result.overflow);

          if (constant_propagation_result.compute_rc)
            ComputeRC0(constant_propagation_result.gpr_value);
        }
      }

      js.fpr_is_store_safe = op.fprIsStoreSafeAfterInst;

      if (!CanMergeNextInstructions(1) || js.op[1].opinfo->type != ::OpType::Integer)
        FlushCarry();

      // If we have a register that will never be used again, discard or flush it.
      if (!bJITRegisterCacheOff)
      {
        gpr.DiscardRegisters(op.gprDiscardable);
        fpr.DiscardRegisters(op.fprDiscardable);
        gpr.DiscardCRRegisters(op.crDiscardable);
      }
      // STV_GC_VOLATIL_x: un volatil muerto en el blr final ya se descarto: no se vacia (ni se convierte).
      const BitSet32 stv_gd = m_stv_volatil ? op.gprDiscardable : BitSet32{};
      const BitSet32 stv_fd = m_stv_volatil ? op.fprDiscardable : BitSet32{};
      const BitSet8 stv_cd = m_stv_volatil ? op.crDiscardable : BitSet8{};  // STV_GC_CRVOL_x
      gpr.FlushRegisters(~(op.gprWillBeRead | op.gprWillBeWritten) & (op.regsIn | op.regsOut) & ~stv_gd,
                         FlushMode::Full);
      fpr.FlushRegisters(~(op.fprWillBeRead | op.fprWillBeWritten) &
                             (op.fregsIn | op.GetFregsOut()) & ~stv_fd,
                         FlushMode::Full);
      gpr.FlushCRRegisters(~(op.crWillBeRead | op.crWillBeWritten) & (op.crIn | op.crOut) & ~stv_cd,
                           FlushMode::Full);
      gpr.FlushRegisters(~op.gprWillBeWritten & op.regsOut & ~stv_gd, FlushMode::Undirty);
      fpr.FlushRegisters(~op.fprWillBeWritten & op.GetFregsOut() & ~stv_fd, FlushMode::Undirty);
      gpr.FlushCRRegisters(~op.crWillBeWritten & op.crOut & ~stv_cd, FlushMode::Undirty);

      if (opinfo->flags & FL_LOADSTORE)
        ++js.numLoadStoreInst;

      if (opinfo->flags & FL_USE_FPU)
        ++js.numFloatingPointInst;
    }

    if (stv_aud)
    {
      stv_aud->push_back({m_code_buffer[i].opinfo->opname, m_code_buffer[i].address, m_code_buffer[i].inst.hex,
                          static_cast<u32>(GetCodePtr() - stv_in),
                          static_cast<u32>(m_far_code.GetCodePtr() - stv_if)});
    }
    // STV_GC_BASE_x: la base deja de valer si alguna instruccion (o las fusionadas) escribe su registro.
    for (int& g : m_stv_base_guest)
    {
      if (g < 0)
        continue;
      for (int k = 0; k <= js.skipInstructions; k++)
      {
        if (m_code_buffer[i + k].regsOut[g])
          g = -1;
        if (g < 0)
          break;
      }
    }
    i += js.skipInstructions;
    js.skipInstructions = 0;
  }

  if (code_block.m_broken)
  {
    gpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
    fpr.Flush(FlushMode::Full, ARM64Reg::INVALID_REG);
    WriteExit(nextPC);
  }

  // STV_GC_SALTOINT_x: saltos internos cuya etiqueta no se alcanzo (el lazo corto antes: gancho de HLE): salen al
  // destino como un bc comun. Llegan con todo en ppcState.
  for (const STVSaltoPendiente& p : m_stv_saltos)
  {
    for (const FixupBranch& f : p.saltos)
      SetJumpTarget(f);
    js.downcountAmount = p.base_dc;
    WriteExit(m_code_buffer[p.destino].address);
  }
  m_stv_saltos.clear();

  // STV_GC_TALON_x: salidas condicionales emitidas fuera de linea, despues del cuerpo del bloque: el camino que sigue de
  // largo solo paga la rama no tomada. Con el downcount y el estado del gather pipe del punto de salida.
  for (const STVTalon& t : m_stv_talones)
  {
    SetJumpTarget(t.rama);
    const u32 dc = js.downcountAmount;
    const u32 fb = js.fifoBytesSinceCheck;
    js.downcountAmount = t.dc;
    js.fifoBytesSinceCheck = 0;
    WriteExit(t.destino);
    js.downcountAmount = dc;
    js.fifoBytesSinceCheck = fb;
  }
  m_stv_talones.clear();

  // STV_GC_AUDITA_x: lo que se emite despues de la ultima instruccion (vaciado y salida de un bloque cortado).
  if (stv_aud && !stv_aud->empty())
  {
    u32 cerca = 0, lejos = 0;
    for (const auto& e : *stv_aud)
    {
      cerca += e.cerca;
      lejos += e.lejos;
    }
    stv_aud->push_back({"<fin>", nextPC, 0, static_cast<u32>(GetCodePtr() - stv_n0) - cerca,
                        static_cast<u32>(m_far_code.GetCodePtr() - stv_f0) - lejos});
  }

  if (HasWriteFailed() || m_far_code.HasWriteFailed())
  {
    if (HasWriteFailed())
      WARN_LOG_FMT(DYNA_REC, "JIT ran out of space in near code region during code generation.");
    if (m_far_code.HasWriteFailed())
      WARN_LOG_FMT(DYNA_REC, "JIT ran out of space in far code region during code generation.");

    return false;
  }

  FlushIcache();
  m_far_code.FlushIcache();

  return true;
}

void JitArm64::LogGeneratedCode() const
{
  std::ostringstream stream;

  stream << "\nPPC Code Buffer:\n";
  for (const PPCAnalyst::CodeOp& op :
       std::span{m_code_buffer.data(), code_block.m_num_instructions})
  {
    fmt::print(stream, "0x{:08x}\t\t{}\n", op.address,
               Common::GekkoDisassembler::Disassemble(op.inst.hex, op.address));
  }

  const JitBlock* const block = js.curBlock;
  stream << "\nHost Near Code:\n";
  m_disassembler->Disassemble(block->normalEntry, block->near_end, stream);
  stream << "\nHost Far Code:\n";
  m_disassembler->Disassemble(block->far_begin, block->far_end, stream);

  // TODO C++20: std::ostringstream::view()
  DEBUG_LOG_FMT(DYNA_REC, "{}", std::move(stream).str());
}
