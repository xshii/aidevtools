import { ArtifactRenderer, ArtifactType, Artifact } from './types';

/**
 * Plugin manager for artifact renderers
 */
export class PluginManager {
    private renderers: Map<ArtifactType, ArtifactRenderer> = new Map();

    /**
     * Register a renderer plugin
     */
    register(renderer: ArtifactRenderer): void {
        this.renderers.set(renderer.type, renderer);
    }

    /**
     * Unregister a renderer plugin
     */
    unregister(type: ArtifactType): void {
        this.renderers.delete(type);
    }

    /**
     * Get a renderer by type
     */
    getRenderer(type: ArtifactType): ArtifactRenderer | undefined {
        return this.renderers.get(type);
    }

    /**
     * Check if a renderer exists
     */
    hasRenderer(type: ArtifactType): boolean {
        return this.renderers.has(type);
    }

    /**
     * Render an artifact
     */
    render(artifact: Artifact): string {
        const renderer = this.renderers.get(artifact.type);
        if (!renderer) {
            return this.renderFallback(artifact);
        }
        return renderer.render(artifact);
    }

    /**
     * Fallback renderer for unknown types
     */
    private renderFallback(artifact: Artifact): string {
        return `
            <div class="fallback">
                <h3>Unknown artifact type: ${artifact.type}</h3>
                <pre>${JSON.stringify(artifact.data, null, 2)}</pre>
            </div>
        `;
    }

    /**
     * List all registered renderer types
     */
    listTypes(): ArtifactType[] {
        return Array.from(this.renderers.keys());
    }
}

// Global plugin manager instance
export const pluginManager = new PluginManager();
