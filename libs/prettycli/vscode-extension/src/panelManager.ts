import * as vscode from 'vscode';
import { Artifact } from './types';
import { pluginManager } from './pluginManager';
import { getBaseHtml } from './renderers/base';
import { sessionManager } from './sessionManager';

/**
 * Manages webview panels for artifact display
 */
export class PanelManager {
    private panels: Map<string, vscode.WebviewPanel> = new Map();
    private panelFiles: Map<string, string> = new Map(); // panelId -> filePath
    private context: vscode.ExtensionContext;

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    /**
     * Show an artifact in a webview panel
     */
    show(artifact: Artifact, panelId?: string): { panelId: string; filePath: string | null } {
        const id = panelId || this.generateId();
        const title = artifact.title || `${artifact.type} artifact`;

        let panel = this.panels.get(id);

        if (!panel) {
            panel = vscode.window.createWebviewPanel(
                'prettycli.artifact',
                title,
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                }
            );

            panel.onDidDispose(() => {
                this.panels.delete(id);
                this.panelFiles.delete(id);
            });

            this.panels.set(id, panel);
        }

        const html = this.renderArtifact(artifact);
        panel.title = title;
        panel.webview.html = html;

        // Save artifact to session folder
        const filePath = sessionManager.saveArtifact(artifact, html);
        if (filePath) {
            this.panelFiles.set(id, filePath);
        }

        return { panelId: id, filePath };
    }

    /**
     * Update an existing panel
     */
    update(panelId: string, artifact: Artifact): { success: boolean; filePath: string | null } {
        const panel = this.panels.get(panelId);
        if (!panel) {
            return { success: false, filePath: null };
        }

        const html = this.renderArtifact(artifact);
        panel.title = artifact.title || panel.title;
        panel.webview.html = html;

        // Save updated artifact
        const filePath = sessionManager.saveArtifact(artifact, html);
        if (filePath) {
            this.panelFiles.set(panelId, filePath);
        }

        return { success: true, filePath };
    }

    /**
     * Get file path for a panel
     */
    getFilePath(panelId: string): string | null {
        return this.panelFiles.get(panelId) || null;
    }

    /**
     * Close a panel
     */
    close(panelId: string): boolean {
        const panel = this.panels.get(panelId);
        if (!panel) {
            return false;
        }

        panel.dispose();
        this.panels.delete(panelId);
        return true;
    }

    /**
     * List all open panels
     */
    list(): string[] {
        return Array.from(this.panels.keys());
    }

    /**
     * Render artifact to HTML
     */
    private renderArtifact(artifact: Artifact): string {
        const content = pluginManager.render(artifact);
        return getBaseHtml(content, artifact.title);
    }

    /**
     * Generate unique panel ID
     */
    private generateId(): string {
        return `panel-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    }

    /**
     * Dispose all panels
     */
    dispose(): void {
        this.panels.forEach(panel => panel.dispose());
        this.panels.clear();
    }
}
