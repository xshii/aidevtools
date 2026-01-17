import { ArtifactRenderer, DiffArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Diff renderer - side by side comparison
 */
export class DiffRenderer implements ArtifactRenderer {
    type = 'diff' as const;

    render(artifact: Artifact): string {
        const diff = artifact as DiffArtifact;
        const { original, modified, originalPath, modifiedPath, language } = diff.data;

        const originalLines = original.split('\n');
        const modifiedLines = modified.split('\n');

        // Simple diff algorithm
        const diffResult = this.computeDiff(originalLines, modifiedLines);

        const leftPanel = this.renderPanel(diffResult.left, 'original');
        const rightPanel = this.renderPanel(diffResult.right, 'modified');

        const langBadge = language ? `<span style="background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 2px 8px; border-radius: 4px; font-size: 11px;">${escapeHtml(language)}</span>` : '';

        return `
            <div>
                ${diff.title ? `<h3>${escapeHtml(diff.title)}</h3>` : ''}
                <div class="diff-container">
                    <div class="diff-panel" style="border: 1px solid var(--border-color); border-radius: 4px; overflow: hidden;">
                        <div class="diff-header" style="display: flex; justify-content: space-between;">
                            <span>${escapeHtml(originalPath || 'Original')}</span>
                            ${langBadge}
                        </div>
                        <div style="overflow-x: auto;">
                            ${leftPanel}
                        </div>
                    </div>
                    <div class="diff-panel" style="border: 1px solid var(--border-color); border-radius: 4px; overflow: hidden;">
                        <div class="diff-header" style="display: flex; justify-content: space-between;">
                            <span>${escapeHtml(modifiedPath || 'Modified')}</span>
                            ${langBadge}
                        </div>
                        <div style="overflow-x: auto;">
                            ${rightPanel}
                        </div>
                    </div>
                </div>
                <div style="margin-top: 8px; color: var(--vscode-descriptionForeground); font-size: 12px;">
                    <span style="color: var(--error-color);">- ${diffResult.removedCount} removed</span>
                    <span style="margin-left: 16px; color: var(--success-color);">+ ${diffResult.addedCount} added</span>
                </div>
            </div>
        `;
    }

    private computeDiff(original: string[], modified: string[]): {
        left: { line: string; type: 'same' | 'removed' | 'empty'; num: number | null }[];
        right: { line: string; type: 'same' | 'added' | 'empty'; num: number | null }[];
        addedCount: number;
        removedCount: number;
    } {
        const left: { line: string; type: 'same' | 'removed' | 'empty'; num: number | null }[] = [];
        const right: { line: string; type: 'same' | 'added' | 'empty'; num: number | null }[] = [];

        // LCS-based diff
        const lcs = this.longestCommonSubsequence(original, modified);

        let oi = 0, mi = 0, li = 0;
        let addedCount = 0, removedCount = 0;

        while (oi < original.length || mi < modified.length) {
            if (li < lcs.length && oi < original.length && original[oi] === lcs[li]) {
                if (mi < modified.length && modified[mi] === lcs[li]) {
                    // Same line
                    left.push({ line: original[oi], type: 'same', num: oi + 1 });
                    right.push({ line: modified[mi], type: 'same', num: mi + 1 });
                    oi++;
                    mi++;
                    li++;
                } else {
                    // Added in modified
                    left.push({ line: '', type: 'empty', num: null });
                    right.push({ line: modified[mi], type: 'added', num: mi + 1 });
                    mi++;
                    addedCount++;
                }
            } else if (oi < original.length && (li >= lcs.length || original[oi] !== lcs[li])) {
                // Removed from original
                left.push({ line: original[oi], type: 'removed', num: oi + 1 });
                right.push({ line: '', type: 'empty', num: null });
                oi++;
                removedCount++;
            } else if (mi < modified.length) {
                // Added in modified
                left.push({ line: '', type: 'empty', num: null });
                right.push({ line: modified[mi], type: 'added', num: mi + 1 });
                mi++;
                addedCount++;
            }
        }

        return { left, right, addedCount, removedCount };
    }

    private longestCommonSubsequence(a: string[], b: string[]): string[] {
        const m = a.length;
        const n = b.length;
        const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));

        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (a[i - 1] === b[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // Backtrack
        const result: string[] = [];
        let i = m, j = n;
        while (i > 0 && j > 0) {
            if (a[i - 1] === b[j - 1]) {
                result.unshift(a[i - 1]);
                i--;
                j--;
            } else if (dp[i - 1][j] > dp[i][j - 1]) {
                i--;
            } else {
                j--;
            }
        }

        return result;
    }

    private renderPanel(
        lines: { line: string; type: string; num: number | null }[],
        side: 'original' | 'modified'
    ): string {
        const rows = lines.map(item => {
            const bgClass = item.type === 'removed' ? 'line-removed' :
                           item.type === 'added' ? 'line-added' : '';
            const lineNum = item.num !== null ? item.num : '';
            const prefix = item.type === 'removed' ? '-' : item.type === 'added' ? '+' : ' ';

            return `<tr class="${bgClass}">
                <td style="user-select: none; padding: 0 8px; text-align: right; color: var(--vscode-editorLineNumber-foreground); border-right: 1px solid var(--border-color); min-width: 40px;">${lineNum}</td>
                <td style="user-select: none; padding: 0 4px; color: ${item.type === 'removed' ? 'var(--error-color)' : item.type === 'added' ? 'var(--success-color)' : 'inherit'};">${prefix}</td>
                <td style="padding: 0 8px; white-space: pre;">${escapeHtml(item.line)}</td>
            </tr>`;
        }).join('');

        return `<table style="border: none; margin: 0; font-family: var(--vscode-editor-font-family); width: 100%;"><tbody>${rows}</tbody></table>`;
    }
}
