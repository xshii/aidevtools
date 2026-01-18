import { ArtifactRenderer, TableArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Table renderer
 */
export class TableRenderer implements ArtifactRenderer {
    type = 'table' as const;

    render(artifact: Artifact): string {
        const table = artifact as TableArtifact;
        const { columns, rows } = table.data;

        const headerCells = columns.map(col => `<th>${escapeHtml(String(col))}</th>`).join('');

        const bodyRows = rows.map(row => {
            const cells = row.map(cell => `<td>${escapeHtml(String(cell))}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');

        return `
            <div>
                ${table.title ? `<h3>${escapeHtml(table.title)}</h3>` : ''}
                <table>
                    <thead>
                        <tr>${headerCells}</tr>
                    </thead>
                    <tbody>
                        ${bodyRows}
                    </tbody>
                </table>
                <div style="margin-top: 8px; color: var(--vscode-descriptionForeground); font-size: 12px;">
                    ${rows.length} rows
                </div>
            </div>
        `;
    }
}
