import { ArtifactRenderer, FileArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * File content renderer
 */
export class FileRenderer implements ArtifactRenderer {
    type = 'file' as const;

    render(artifact: Artifact): string {
        const file = artifact as FileArtifact;
        const { path, content, language, startLine, endLine } = file.data;

        const lines = content?.split('\n') || [];
        const start = startLine || 1;
        const end = endLine || lines.length;

        const displayLines = lines.slice(start - 1, end);
        const lineNumbers = displayLines.map((_, i) => start + i);

        const codeContent = displayLines.map((line, i) => {
            const lineNum = lineNumbers[i];
            return `<tr>
                <td style="user-select: none; padding: 0 12px; text-align: right; color: var(--vscode-editorLineNumber-foreground); border-right: 1px solid var(--border-color);">${lineNum}</td>
                <td style="padding: 0 12px; white-space: pre;">${escapeHtml(line)}</td>
            </tr>`;
        }).join('');

        const fileName = path.split('/').pop() || path;
        const langBadge = language ? `<span style="background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 2px 8px; border-radius: 4px; font-size: 11px;">${escapeHtml(language)}</span>` : '';

        return `
            <div class="file-content">
                <div class="file-header">
                    <span class="file-path">${escapeHtml(path)}</span>
                    ${langBadge}
                </div>
                <div style="overflow-x: auto;">
                    <table style="border: none; margin: 0; font-family: var(--vscode-editor-font-family);">
                        <tbody>
                            ${codeContent}
                        </tbody>
                    </table>
                </div>
                ${startLine || endLine ? `<div style="padding: 8px; color: var(--vscode-descriptionForeground); font-size: 12px; border-top: 1px solid var(--border-color);">Lines ${start}-${end}</div>` : ''}
            </div>
        `;
    }
}
