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

from .analyzer import (
    AnalysisSummary,
    PaperAnalyzer,
)
from .chip import (
    ChipSpec,
    ComputeUnitSpec,
    MemoryLevelSpec,
    MemorySpec,
    PipelineSpec,
    VectorUnitSpec,
    list_chips,
    load_chip_spec,
)
from .export import (
    export_csv,
    export_json,
    export_xlsx,
)
from .latency import (
    GanttData,
    GanttItem,
    LatencyBreakdown,
    LatencyResult,
)
from .models import (
    MODEL_CONFIGS,
    bert_layer,
    from_preset,
    gpt2_layer,
    list_presets,
    llama_layer,
    transformer_layer,
    vit_layer,
)
from .passes import (
    ALL_PASSES,
    BackwardPrefetchPass,
    BasePass,
    CubeVectorParallelPass,
    ForwardPrefetchPass,
    MemoryEfficiencyPass,
    OverheadPass,
    PassConfig,
    PassPreset,
    PassResult,
    RooflinePass,
)
from .profile import (
    OpProfile,
    dtype_bytes,
)

__all__ = [
    # Profile
    "OpProfile",
    "dtype_bytes",
    # Chip
    "ChipSpec",
    "ComputeUnitSpec",
    "VectorUnitSpec",
    "MemorySpec",
    "MemoryLevelSpec",
    "PipelineSpec",
    "load_chip_spec",
    "list_chips",
    # Latency
    "LatencyResult",
    "LatencyBreakdown",
    "GanttItem",
    "GanttData",
    # Analyzer
    "PaperAnalyzer",
    "AnalysisSummary",
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
