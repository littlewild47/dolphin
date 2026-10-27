// Copyright 2014 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include <vector>

#include <cstddef>
#include <map>
#include <optional>

#include "Common/Arm64Emitter.h"
#include "Common/RangeSizeSet.h"

#include "Core/PowerPC/JitArm64/JitArm64Cache.h"
#include "Core/PowerPC/JitArm64/JitArm64_RegCache.h"
#include "Core/PowerPC/JitArmCommon/BackPatch.h"
#include "Core/PowerPC/JitCommon/ConstantPropagation.h"
#include "Core/PowerPC/JitCommon/JitAsmCommon.h"
#include "Core/PowerPC/JitCommon/JitBase.h"
#include "Core/PowerPC/PPCAnalyst.h"

class HostDisassembler;

// STV_GC_AUDITA_x (instrumento, ver Jit.cpp)
struct STVAuditaOp
{
  const char* nombre;
  u32 pc;
  u32 hex;
  u32 cerca;
  u32 lejos;
};
extern bool g_stv_audita;
extern std::unordered_map<u32, std::vector<STVAuditaOp>> g_stv_audita_bloques;

// STV_GC_IC_x
struct STVICSitioJit
{
  u32 destino = 0;
  u32 racha = 0;
  u32 llamadas = 0;
  u8 estado = 0;  // 0 aprendiendo, 1 adivinado, 2 mega
};
extern std::unordered_map<u32, STVICSitioJit> g_stv_ic_sitios;
u32 STVICAprender(u32 sitio, u32 destino);
// STV_GC_ICCUENTA_x
void STVICRegistrar(u32 sitio, u32 destino);
void STVICVolcar(const char* ruta);

class JitArm64 : public JitBase, public Arm64Gen::ARM64CodeBlock, public CommonAsmRoutinesBase
{
public:
  explicit JitArm64(Core::System& system);
  JitArm64(const JitArm64&) = delete;
  JitArm64(JitArm64&&) = delete;
  JitArm64& operator=(const JitArm64&) = delete;
  JitArm64& operator=(JitArm64&&) = delete;
  ~JitArm64() override;

  void Init() override;
  void Shutdown() override;

  JitCommon::ConstantPropagation& GetConstantPropagation() { return m_constant_propagation; }

  JitBaseBlockCache* GetBlockCache() override { return &blocks; }
  bool IsInCodeSpace(const u8* ptr) const { return IsInSpace(ptr); }
  bool HandleFault(uintptr_t access_address, SContext* ctx) override;
  void DoBacktrace(uintptr_t access_address, SContext* ctx);
  bool HandleFastmemFault(SContext* ctx);

  void ClearCache() override;

  CommonAsmRoutinesBase* GetAsmRoutines() override { return this; }
  void Run() override;
  void SingleStep() override;

  void Jit(u32 em_address) override;
  void Jit(u32 em_address, bool clear_cache_and_retry_on_failure);

  void EraseSingleBlock(const JitBlock& block) override;
  std::vector<MemoryStats> GetMemoryStats() const override;

  std::size_t DisassembleNearCode(const JitBlock& block, std::ostream& stream) const override;
  std::size_t DisassembleFarCode(const JitBlock& block, std::ostream& stream) const override;

  const char* GetName() const override { return "JITARM64"; }

  // STV_GC_JITDC_v1: con la valvula STVDowncountReg prendida, el contador de ciclos vive en
  // DOWNCOUNT_REG (W27) dentro del codigo JIT. Invariante: siempre que el control esta en C++,
  // ppcState.downcount == W27. Por eso toda llamada del codigo JIT a C++ pasa por estos envoltorios, que tapan
  // a los de ARM64XEmitter: STR W27 antes, LDR W27 despues (el C++ puede leer o reescribir el downcount:
  // ForceExceptionCheck, Idle, GetTicks). Con la valvula apagada emiten exactamente lo mismo que antes.
  // Las rutinas del despachador (JitAsm.cpp) NO usan estos envoltorios: llaman a la version de ARM64XEmitter y
  // sincronizan a mano con STVDcGuardar/STVDcCargar(true), que emiten NOP con la valvula apagada para que la
  // disposicion sea identica en los dos modos (ver JitAsm.cpp).
  bool STVDcReg() const { return m_stv_dc_reg; }
  void STVDcGuardar(bool nop_si_apagada = false);
  void STVDcCargar(bool nop_si_apagada = false);

  // STV_GC_JITGF_v1: con la valvula STVGprFijos (STVGf() != 0), r3,r0,r31,r4,r1,r30 viven en
  // W19-W24 en todo el codigo JIT (ver JitArm64_RegCache.h). Invariante: la copia en ppcState.gpr solo vale con el
  // control en C++ en un punto de sincronizacion. STVGfGuardar: los seis registros -> ppcState (3 STP);
  // STVGfCargar: ppcState -> los seis (3 LDP). Con la valvula apagada no emiten nada, o 3 NOP si se pide (rutinas del
  // despachador: misma disposicion en los tres modos, como STVDc*). A diferencia del downcount, los envoltorios
  // ABI_CallFunction NO sincronizan los fijos: ningun C++ llamado desde un bloque lee ni escribe ppcState.gpr salvo
  // el interprete (FallBackToInterpreter), HLE y los puntos de parada, que sincronizan a mano; los demas (memoria
  // lenta, MMIO, gather pipe, excepciones, BranchWatch, Idle, perfilado, invalidacion de icache) no tocan gpr, y
  // los callee-saved sobreviven a la llamada por ABI. Los puntos de vigilancia si leen gpr: con jo.memcheck la
  // valvula se fuerza a 0.
  void STVGfGuardar(bool nop_si_apagada = false);
  void STVGfCargar(bool nop_si_apagada = false);
  // STV_GC_JITGF_v1: modo 2, al entrar a cada bloque: cada fijo contra su copia en ppcState; si difieren, BRK #0x5A.
  void STVGfVerificar();

  template <typename FuncRet, typename... FuncArgs, typename... Args>
  void ABI_CallFunction(FuncRet (*func)(FuncArgs...), Args... args)
  {
    STVDcGuardar();
    Arm64Gen::ARM64XEmitter::ABI_CallFunction(func, args...);
    STVDcCargar();
  }

  template <typename FuncRet, typename... FuncArgs, typename... Args>
  void ABI_CallLambdaFunction(const std::function<FuncRet(FuncArgs...)>* f, Args... args)
  {
    STVDcGuardar();
    Arm64Gen::ARM64XEmitter::ABI_CallLambdaFunction(f, args...);
    STVDcCargar();
  }

  // OPCODES
  using Instruction = void (JitArm64::*)(UGeckoInstruction);
  void FallBackToInterpreter(UGeckoInstruction inst);
  void DoNothing(UGeckoInstruction inst);
  void HLEFunction(u32 hook_index);

  void DynaRunTable4(UGeckoInstruction inst);
  void DynaRunTable19(UGeckoInstruction inst);
  void DynaRunTable31(UGeckoInstruction inst);
  void DynaRunTable59(UGeckoInstruction inst);
  void DynaRunTable63(UGeckoInstruction inst);

  // Force break
  void Break(UGeckoInstruction inst);

  // Branch
  void sc(UGeckoInstruction inst);
  void rfi(UGeckoInstruction inst);
  void bx(UGeckoInstruction inst);
  void bcx(UGeckoInstruction inst);
  void bcctrx(UGeckoInstruction inst);
  void bclrx(UGeckoInstruction inst);

  // Integer
  void arith_imm(UGeckoInstruction inst);
  void boolX(UGeckoInstruction inst);
  void addx(UGeckoInstruction inst);
  void addix(UGeckoInstruction inst);
  void extsXx(UGeckoInstruction inst);
  void cntlzwx(UGeckoInstruction inst);
  void negx(UGeckoInstruction inst);
  void cmp(UGeckoInstruction inst);
  void cmpl(UGeckoInstruction inst);
  void cmpi(UGeckoInstruction inst);
  void cmpli(UGeckoInstruction inst);
  void rlwinmx(UGeckoInstruction inst);
  void rlwnmx(UGeckoInstruction inst);
  void srawix(UGeckoInstruction inst);
  void mullwx(UGeckoInstruction inst);
  void mulhwx(UGeckoInstruction inst);
  void mulhwux(UGeckoInstruction inst);
  void addic(UGeckoInstruction inst);
  void mulli(UGeckoInstruction inst);
  void addzex(UGeckoInstruction inst);
  void divwx(UGeckoInstruction inst);
  void subfx(UGeckoInstruction inst);
  void addcx(UGeckoInstruction inst);
  void slwx(UGeckoInstruction inst);
  void srwx(UGeckoInstruction inst);
  void srawx(UGeckoInstruction inst);
  void rlwimix(UGeckoInstruction inst);
  void subfex(UGeckoInstruction inst);
  void subfzex(UGeckoInstruction inst);
  void subfic(UGeckoInstruction inst);
  void addex(UGeckoInstruction inst);
  void divwux(UGeckoInstruction inst);

  // System Registers
  void mtmsr(UGeckoInstruction inst);
  void mfmsr(UGeckoInstruction inst);
  void mcrf(UGeckoInstruction inst);
  void mcrxr(UGeckoInstruction inst);
  void mfsr(UGeckoInstruction inst);
  void mfsrin(UGeckoInstruction inst);
  void twx(UGeckoInstruction inst);
  void mfspr(UGeckoInstruction inst);
  void mftb(UGeckoInstruction inst);
  void mtspr(UGeckoInstruction inst);
  void crXXX(UGeckoInstruction inst);
  void mfcr(UGeckoInstruction inst);
  void mtcrf(UGeckoInstruction inst);
  void mcrfs(UGeckoInstruction inst);
  void mffsx(UGeckoInstruction inst);
  void mtfsb0x(UGeckoInstruction inst);
  void mtfsb1x(UGeckoInstruction inst);
  void mtfsfix(UGeckoInstruction inst);
  void mtfsfx(UGeckoInstruction inst);

  // LoadStore
  void lXX(UGeckoInstruction inst);
  void stX(UGeckoInstruction inst);
  void lmw(UGeckoInstruction inst);
  void stmw(UGeckoInstruction inst);
  void dcbx(UGeckoInstruction inst);
  void dcbt(UGeckoInstruction inst);
  void dcbz(UGeckoInstruction inst);
  void eieio(UGeckoInstruction inst);

  // LoadStore floating point
  void lfXX(UGeckoInstruction inst);
  void stfXX(UGeckoInstruction inst);

  // Floating point
  void fp_arith(UGeckoInstruction inst);
  void fp_logic(UGeckoInstruction inst);
  void fselx(UGeckoInstruction inst);
  void fcmpX(UGeckoInstruction inst);
  void frspx(UGeckoInstruction inst);
  void fctiwx(UGeckoInstruction inst);
  void fresx(UGeckoInstruction inst);
  void frsqrtex(UGeckoInstruction inst);

  // Paired
  void ps_mergeXX(UGeckoInstruction inst);
  void ps_arith(UGeckoInstruction inst);
  void ps_sel(UGeckoInstruction inst);
  void ps_sumX(UGeckoInstruction inst);
  void ps_res(UGeckoInstruction inst);
  void ps_rsqrte(UGeckoInstruction inst);
  void ps_cmpXX(UGeckoInstruction inst);

  // Loadstore paired
  void psq_lXX(UGeckoInstruction inst);
  void psq_stXX(UGeckoInstruction inst);

  void ConvertDoubleToSingleLower(size_t guest_reg, Arm64Gen::ARM64Reg dest_reg,
                                  Arm64Gen::ARM64Reg src_reg);
  void ConvertDoubleToSinglePair(size_t guest_reg, Arm64Gen::ARM64Reg dest_reg,
                                 Arm64Gen::ARM64Reg src_reg);
  void ConvertSingleToDoubleLower(size_t guest_reg, Arm64Gen::ARM64Reg dest_reg,
                                  Arm64Gen::ARM64Reg src_reg,
                                  Arm64Gen::ARM64Reg scratch_reg = Arm64Gen::ARM64Reg::INVALID_REG);
  void ConvertSingleToDoublePair(size_t guest_reg, Arm64Gen::ARM64Reg dest_reg,
                                 Arm64Gen::ARM64Reg src_reg,
                                 Arm64Gen::ARM64Reg scratch_reg = Arm64Gen::ARM64Reg::INVALID_REG);

  void FloatCompare(UGeckoInstruction inst, bool upper = false);

  bool IsFPRStoreSafe(size_t guest_reg) const;

  void rlwinmx_internal(UGeckoInstruction inst, u32 sh);

protected:
  struct FastmemArea
  {
    const u8* fast_access_code;
    const u8* slow_access_code;
  };

  void SetBlockLinkingEnabled(bool enabled);
  void SetOptimizationEnabled(bool enabled);

  void CompileInstruction(PPCAnalyst::CodeOp& op);

  bool HandleFunctionHooking(u32 address);

  // Simple functions to switch between near and far code emitting
  void SwitchToFarCode()
  {
    m_near_code = GetWritableCodePtr();
    m_near_code_end = GetWritableCodeEnd();
    m_near_code_write_failed = HasWriteFailed();
    SetCodePtrUnsafe(m_far_code.GetWritableCodePtr(), m_far_code.GetWritableCodeEnd(),
                     m_far_code.HasWriteFailed());
    AlignCode16();
    m_in_far_code = true;
  }

  void SwitchToNearCode()
  {
    m_far_code.SetCodePtrUnsafe(GetWritableCodePtr(), GetWritableCodeEnd(), HasWriteFailed());
    SetCodePtrUnsafe(m_near_code, m_near_code_end, m_near_code_write_failed);
    m_in_far_code = false;
  }

  bool IsInFarCode() const { return m_in_far_code; }

  // Dump a memory range of code
  void DumpCode(const u8* start, const u8* end);

  // This enum is used for selecting an implementation of EmitBackpatchRoutine.
  enum class MemAccessMode
  {
    // Always calls the slow C++ code. For performance reasons, should generally only be used if
    // the guest address is known in advance and IsOptimizableRAMAddress returns false for it.
    AlwaysSlowAccess,
    // Only emits fast access code. Must only be used if the guest address is known in advance
    // and IsOptimizableRAMAddress returns true for it, otherwise Dolphin will likely crash!
    AlwaysFastAccess,
    // Best in most cases. If backpatching is possible (!emitting_routine && jo.fastmem):
    // Tries to run fast access code, and if that fails, uses backpatching to replace the code
    // with a call to the slow C++ code. Otherwise: Checks whether the fast access code will work,
    // then branches to either the fast access code or the slow C++ code.
    Auto,
  };

  // This is the core routine for accessing emulated memory, with support for
  // many different kinds of loads and stores as well as fastmem/backpatching.
  //
  // Registers used:
  //
  //                 addr     scratch
  // Store:          X2       X1
  // Load:           X1
  // Zero 256:       X1       X30
  // Store float:    X2       Q0
  // Load float:     X1
  //
  // If mode == AlwaysFastAccess, the addr argument can be any register.
  // Otherwise it must be the register listed in the table above.
  //
  // Additional scratch registers are used in the following situations:
  //
  // emitting_routine && mode == Auto:                                            X0
  // emitting_routine && mode == Auto && !(flags & BackPatchInfo::FLAG_STORE):    X3
  // emitting_routine && mode != AlwaysSlowAccess && !jo.fastmem:                 X3
  // mode != AlwaysSlowAccess && !jo.fastmem:                                     X0
  // !emitting_routine && mode != AlwaysFastAccess && jo.memcheck &&
  //         (flags & BackPatchInfo::FLAG_LOAD):                                  X0
  // !emitting_routine && mode != AlwaysSlowAccess && !jo.fastmem:                X30
  // !emitting_routine && mode == Auto && jo.fastmem:                             X30
  //
  // Furthermore:
  // - Any callee-saved register which isn't marked in gprs_to_push/fprs_to_push may be
  //   clobbered if mode != AlwaysFastAccess.
  // - If !emitting_routine && mode != AlwaysFastAccess && jo.memcheck, X30 must not
  //   contain a guest register.
  void EmitBackpatchRoutine(u32 flags, MemAccessMode mode, Arm64Gen::ARM64Reg RS,
                            Arm64Gen::ARM64Reg addr, BitSet32 gprs_to_push = BitSet32(0),
                            BitSet32 fprs_to_push = BitSet32(0), bool emitting_routine = false);

  // temp_gpr must be a valid register, but temp_fpr can be INVALID_REG.
  void FlushPPCStateBeforeSlowAccess(Arm64Gen::ARM64Reg temp_gpr, Arm64Gen::ARM64Reg temp_fpr);

  // Loadstore routines
  void SafeLoadToReg(u32 dest, s32 addr, s32 offsetReg, u32 flags, s32 offset, bool update);
  void SafeStoreFromReg(s32 dest, u32 value, s32 regOffset, u32 flags, s32 offset, bool update);
  // If lookup succeeds, writes upper 15 bits of physical address to addr_out. If not,
  // jumps to the returned FixupBranch. Clobbers tmp and the 17 lower bits of addr_out.
  Arm64Gen::FixupBranch BATAddressLookup(Arm64Gen::ARM64Reg addr_out, Arm64Gen::ARM64Reg addr_in,
                                         Arm64Gen::ARM64Reg tmp, const void* bat_table);
  Arm64Gen::FixupBranch CheckIfSafeAddress(Arm64Gen::ARM64Reg addr, Arm64Gen::ARM64Reg tmp1,
                                           Arm64Gen::ARM64Reg tmp2);

  bool DoJit(u32 em_address, JitBlock* b, u32 nextPC);

  void Trace();

  // Finds a free memory region and sets the near and far code emitters to point at that region.
  // On success, returns the index of the memory region (either 0 or 1).
  // If either near code or far code is full, returns std::nullopt.
  std::optional<size_t> SetEmitterStateToFreeCodeRegion();

  void DoDownCount();
  void Cleanup();
  void ResetStack();

  void FreeRanges();
  void GenerateAsmAndResetFreeMemoryRanges();
  void ResetFreeMemoryRanges(size_t routines_near_size, size_t routines_far_size);

  void IntializeSpeculativeConstants();

  void LogGeneratedCode() const;

  // AsmRoutines
  void GenerateAsm();
  void GenerateCommonAsm();
  void GenerateFres();
  void GenerateFrsqrte();
  void GenerateConvertDoubleToSingle();
  void GenerateConvertSingleToDouble();
  void GenerateFPRF(bool single);
  void GenerateFmaddsEft();
  void GeneratePsMaddEft();
  void GenerateQuantizedLoads();
  void GenerateQuantizedStores();

  void EmitUpdateMembase();
  void MSRUpdated(u32 msr);
  void MSRUpdated(Arm64Gen::ARM64Reg msr);

  // Branch Watch
  template <bool condition>
  void WriteBranchWatch(u32 origin, u32 destination, UGeckoInstruction inst,
                        BitSet32 gpr_caller_save, BitSet32 fpr_caller_save);
  void WriteBranchWatchDestInRegister(u32 origin, Arm64Gen::ARM64Reg destination,
                                      UGeckoInstruction inst, BitSet32 gpr_caller_save,
                                      BitSet32 fpr_caller_save);

  // Exits
  void
  WriteExit(u32 destination, bool LK = false, u32 exit_address_after_return = 0,
            Arm64Gen::ARM64Reg exit_address_after_return_reg = Arm64Gen::ARM64Reg::INVALID_REG);
  void
  WriteExit(Arm64Gen::ARM64Reg dest, bool LK = false, u32 exit_address_after_return = 0,
            Arm64Gen::ARM64Reg exit_address_after_return_reg = Arm64Gen::ARM64Reg::INVALID_REG);
  void WriteExceptionExit(u32 destination, bool only_external = false,
                          bool always_exception = false);
  void WriteExceptionExit(Arm64Gen::ARM64Reg dest, bool only_external = false,
                          bool always_exception = false);
  void WriteConditionalExceptionExit(int exception, u64 increment_sp_on_exit = 0);
  void WriteConditionalExceptionExit(int exception, Arm64Gen::ARM64Reg temp_gpr,
                                     Arm64Gen::ARM64Reg temp_fpr = Arm64Gen::ARM64Reg::INVALID_REG,
                                     u64 increment_sp_on_exit = 0);
  void
  FakeLKExit(u32 exit_address_after_return,
             Arm64Gen::ARM64Reg exit_address_after_return_reg = Arm64Gen::ARM64Reg::INVALID_REG);
  void WriteBLRExit(Arm64Gen::ARM64Reg dest);

  void GetCRFieldBit(int field, int bit, Arm64Gen::ARM64Reg out);
  // This assumes that all bits except for bit 0 (LSB) are set to 0. But if bits_1_to_31_are_set
  // equals true, it instead assumes that all of bits 1 to 31 are set.
  void SetCRFieldBit(int field, int bit, Arm64Gen::ARM64Reg in, bool negate = false,
                     bool bits_1_to_31_are_set = false);
  void ClearCRFieldBit(int field, int bit);
  void SetCRFieldBit(int field, int bit);
  void FixGTBeforeSettingCRFieldBit(Arm64Gen::ARM64Reg reg);
  Arm64Gen::FixupBranch JumpIfCRFieldBit(int field, int bit, bool jump_if_set);

  void UpdateFPExceptionSummary(Arm64Gen::ARM64Reg fpscr);
  void UpdateRoundingMode();

  void ComputeRC0(Arm64Gen::ARM64Reg reg);
  void ComputeRC0(u32 imm);
  void GenerateConstantOverflow(bool overflow);
  void ComputeCarry(Arm64Gen::ARM64Reg reg);  // reg must contain 0 or 1
  void ComputeCarry(bool carry);
  void ComputeCarry();
  void LoadCarry();
  void FlushCarry();

  // STV_GC_SALTOINT_x: saltos internos hacia adelante (bc a una instruccion posterior del mismo bloque). El camino
  // tomado vacia en MaintainState y salta a la etiqueta; en la etiqueta el camino de largo vacia en Full, asi los dos
  // llegan con todo en ppcState (los fijos en su registro). El downcount queda en la base del primer salto: el de largo
  // descuenta lo suyo en la etiqueta, y un salto posterior descuenta su diferencia antes de saltar.
  struct STVSaltoPendiente
  {
    u32 destino;
    std::vector<Arm64Gen::FixupBranch> saltos;
    u32 base_dc;
    bool fp;
    u32 fifo;
    bool fifo_obligado;
  };
  std::vector<STVSaltoPendiente> m_stv_saltos;
  bool STVSaltoInterno(Arm64Gen::FixupBranch* unica, Arm64Gen::ARM64Reg tmp);
  static Arm64Gen::FixupBranch STVInvertir(Arm64Gen::FixupBranch f);
  // STV_GC_TALON_x (bit 32768)
  struct STVTalon
  {
    Arm64Gen::FixupBranch rama;
    u32 destino;
    u32 dc;
  };
  std::vector<STVTalon> m_stv_talones;
  // STV_GC_CONVTALON_x (bit 262144): el resto del chequeo de una conversion simple -> doble (ceros y camino exacto) va a
  // un talon al final del bloque; en linea queda el chequeo rapido con la rama invertida y la conversion.
  struct STVConvTalon
  {
    Arm64Gen::FixupBranch rama;
    Arm64Gen::ARM64Reg dest;
    Arm64Gen::ARM64Reg src;
    Arm64Gen::ARM64Reg scratch;
    BitSet32 gpr_saved;
    bool par;
    const u8* vuelta;
  };
  std::vector<STVConvTalon> m_stv_conv_talones;
  bool m_stv_conv_cerrado = false;
  bool STVConvTalonPosible() const;
  void STVEmitirConvTalones();
  void STVEtiqueta(u32 indice);
  void STVRestarDowncount(u32 ciclos);
  // STV_GC_CORTO_x (bit 16 de la valvula): salida enlazada sin chequeo de downcount si va hacia adelante. Todo ciclo del
  // grafo de bloques tiene una arista a una direccion <= la de su bloque de origen (o pasa por el despachador), y ahi
  // se sigue chequeando; el downcount se descuenta igual en todas.
  // STV_GC_BASE_x (bit 16384): X25 = MEM_REG + (u64)rA del invitado m_stv_base_guest (-1: nada). Un acceso con la misma
  // base va como [X25, #desplazamiento] (las guardas de 2 GiB de la arena cubren el desborde de 32 bits, como en Jit64).
  // STV_GC_BASE2_x (bit 65536): segunda ranura en X26 (W26 sale del pool; solo lo escriben las salidas al irse).
  std::array<int, 2> m_stv_base_guest{-1, -1};
  u32 m_stv_base_lru = 0;  // ranura usada mas recientemente
  Arm64Gen::ARM64Reg m_stv_base_reg = Arm64Gen::ARM64Reg::X25;
  bool m_stv_base_usar = false;
  s32 m_stv_base_off = 0;
  bool STVBasePreparar(int guest, Arm64Gen::ARM64Reg host, s32 off, u32 bytes, bool update);
  // STV_GC_LLAMADA_x
  bool m_stv_lr_en_salida = false;
  bool m_stv_blr_sin_and = false;
  bool STVCorto(u32 destino) const { return (m_stv_volatil & 16) && destino > js.blockStart; }

  void reg_imm(u32 d, u32 a, u32 value,
               void (ARM64XEmitter::*op)(Arm64Gen::ARM64Reg, Arm64Gen::ARM64Reg, u64,
                                         Arm64Gen::ARM64Reg),
               bool Rc = false);
  bool MultiplyImmediate(u32 imm, int a, int d, bool rc);

  void SetFPRFIfNeeded(bool single, Arm64Gen::ARM64Reg reg);
  void Force25BitPrecision(Arm64Gen::ARM64Reg output, Arm64Gen::ARM64Reg input);

  // <Fast path fault location, slow path handler location>
  std::map<const u8*, FastmemArea> m_fault_to_handler{};
  Arm64GPRCache gpr;
  Arm64FPRCache fpr;

  JitCommon::ConstantPropagation m_constant_propagation;

  JitArm64BlockCache blocks{*this};

  Arm64Gen::ARM64FloatEmitter m_float_emit;

  // Because B instructions can't jump farther than +/- 128 MiB, code memory is allocated like this:
  //
  // m_far_code_0: x MiB of unused space, followed by 64 - x MiB of far code
  // m_near_code_0: 64 MiB of near code
  // m_near_code_1: x MiB of asm routines, followed by 64 - x MiB of near code
  // m_far_code_1: 64 MiB of far code
  //
  // This ensures that:
  //
  // * Any code in m_near_code_0 can reach any code in m_far_code_0, and vice versa
  // * Any code in m_near_code_1 can reach any code in m_far_code_1, and vice versa
  // * Any near code can reach any near code
  // * Any code can reach any asm routine
  //
  // m_far_code_0 and m_far_code_1 can't reach each other, but that isn't needed, because all blocks
  // have their entry points in near code.

  Arm64Gen::ARM64CodeBlock m_near_code_0;
  Arm64Gen::ARM64CodeBlock m_near_code_1;
  Arm64Gen::ARM64CodeBlock m_far_code_0;
  Arm64Gen::ARM64CodeBlock m_far_code_1;

  Arm64Gen::ARM64CodeBlock m_far_code;
  bool m_in_far_code = false;

  // Backed up when we switch to far code.
  u8* m_near_code = nullptr;
  u8* m_near_code_end = nullptr;
  bool m_near_code_write_failed = false;

  Common::RangeSizeSet<u8*> m_free_ranges_near_0;
  Common::RangeSizeSet<u8*> m_free_ranges_near_1;
  Common::RangeSizeSet<u8*> m_free_ranges_far_0;
  Common::RangeSizeSet<u8*> m_free_ranges_far_1;

  std::unique_ptr<HostDisassembler> m_disassembler;
};
