"""Paper Analysis 模块

性能分析工具，用于分析模型在不同芯片上的时延、带宽、算力表现。

Usage:
    from aidevtools.analysis import PaperAnalyzer, PassConfig, PassPreset

    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profile(profile)
    result = analyzer.analyze()
    analyzer.print_summary()

    from aidevtools.analysis import export_xlsx
    export_xlsx(result, "report.xlsx")
"""

from .profile import (
    OpProfile,
    profile_matmul,
    profile_layernorm,
    profile_softmax,
    profile_transpose,
    profile_attention,
    profile_gelu,
    profile_add,
    dtype_bytes,
)

from .chip import (
    ChipSpec,
    ComputeUnitSpec,
    VectorUnitSpec,
    MemorySpec,
    MemoryLevelSpec,
    PipelineSpec,
    load_chip_spec,
    get_chip_spec,
    list_chips,
)

from .latency import (
    LatencyResult,
    LatencyBreakdown,
    GanttItem,
    GanttData,
)

from .analyzer import (
    PaperAnalyzer,
    AnalysisSummary,
    AnalysisMode,
)

from .passes import (
    PassConfig,
    PassResult,
    PassPreset,
    BasePass,
    RooflinePass,
    MemoryEfficiencyPass,
    ForwardPrefetchPass,
    BackwardPrefetchPass,
    CubeVectorParallelPass,
    OverheadPass,
    ALL_PASSES,
)

from .export import (
    export_xlsx,
    export_csv,
    export_json,
)

from .models import (
    transformer_layer,
    llama_layer,
    gpt2_layer,
    bert_layer,
    vit_layer,
    from_preset,
    list_presets,
    MODEL_CONFIGS,
)


__all__ = [
    # Profile
    "OpProfile",
    "profile_matmul",
    "profile_layernorm",
    "profile_softmax",
    "profile_transpose",
    "profile_attention",
    "profile_gelu",
    "profile_add",
    "dtype_bytes",
    # Chip
    "ChipSpec",
    "ComputeUnitSpec",
    "VectorUnitSpec",
    "MemorySpec",
    "MemoryLevelSpec",
    "PipelineSpec",
    "load_chip_spec",
    "get_chip_spec",
    "list_chips",
    # Latency
    "LatencyResult",
    "LatencyBreakdown",
    "GanttItem",
    "GanttData",
    # Analyzer
    "PaperAnalyzer",
    "AnalysisSummary",
    "AnalysisMode",
    # Passes
    "PassConfig",
    "PassResult",
    "PassPreset",
    "BasePass",
    "RooflinePass",
    "MemoryEfficiencyPass",
    "ForwardPrefetchPass",
    "BackwardPrefetchPass",
    "CubeVectorParallelPass",
    "OverheadPass",
    "ALL_PASSES",
    # Export
    "export_xlsx",
    "export_csv",
    "export_json",
    # Models
    "transformer_layer",
    "llama_layer",
    "gpt2_layer",
    "bert_layer",
    "vit_layer",
    "from_preset",
    "list_presets",
    "MODEL_CONFIGS",
]
