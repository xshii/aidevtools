/**
 * Artifact types supported by prettycli
 */
export type ArtifactType = 'chart' | 'table' | 'file' | 'diff' | 'web' | 'markdown' | 'json' | 'image' | 'csv' | 'xlsx';

/**
 * Base artifact interface
 */
export interface Artifact {
    type: ArtifactType;
    title?: string;
    data: unknown;
}

/**
 * Chart artifact
 */
export interface ChartArtifact extends Artifact {
    type: 'chart';
    data: {
        chartType: 'bar' | 'line' | 'pie' | 'scatter';
        labels: string[];
        datasets: {
            label: string;
            data: number[];
            color?: string;
        }[];
    };
}

/**
 * Table artifact
 */
export interface TableArtifact extends Artifact {
    type: 'table';
    data: {
        columns: string[];
        rows: (string | number)[][];
    };
}

/**
 * File artifact
 */
export interface FileArtifact extends Artifact {
    type: 'file';
    data: {
        path: string;
        content?: string;
        language?: string;
        startLine?: number;
        endLine?: number;
    };
}

/**
 * Diff artifact
 */
export interface DiffArtifact extends Artifact {
    type: 'diff';
    data: {
        original: string;
        modified: string;
        originalPath?: string;
        modifiedPath?: string;
        language?: string;
    };
}

/**
 * Web artifact
 */
export interface WebArtifact extends Artifact {
    type: 'web';
    data: {
        html?: string;
        url?: string;
    };
}

/**
 * Markdown artifact
 */
export interface MarkdownArtifact extends Artifact {
    type: 'markdown';
    data: {
        content: string;
    };
}

/**
 * JSON artifact
 */
export interface JsonArtifact extends Artifact {
    type: 'json';
    data: {
        content: unknown;
        collapsed?: boolean;
    };
}

/**
 * API message format
 */
export interface ApiMessage {
    id: string;
    action: 'render' | 'close' | 'update' | 'list';
    artifact?: Artifact;
    panelId?: string;
}

/**
 * API response format
 */
export interface ApiResponse {
    id: string;
    success: boolean;
    error?: string;
    data?: unknown;
}

/**
 * Renderer interface for plugins
 */
export interface ArtifactRenderer {
    type: ArtifactType;
    render(artifact: Artifact): string;
}
