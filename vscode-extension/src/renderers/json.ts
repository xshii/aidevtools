import { ArtifactRenderer, JsonArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * JSON renderer with syntax highlighting and collapsible nodes
 */
export class JsonRenderer implements ArtifactRenderer {
    type = 'json' as const;

    render(artifact: Artifact): string {
        const json = artifact as JsonArtifact;
        const { content, collapsed } = json.data;

        const html = this.renderValue(content, 0, collapsed ?? false);

        return `
            <div>
                ${json.title ? `<h3>${escapeHtml(json.title)}</h3>` : ''}
                <pre style="margin: 0; padding: 12px; background: var(--vscode-textCodeBlock-background); border-radius: 4px; overflow-x: auto;">${html}</pre>
            </div>
            <script>
                document.querySelectorAll('.collapsible').forEach(el => {
                    el.addEventListener('click', (e) => {
                        e.target.classList.toggle('collapsed');
                    });
                });
            </script>
        `;
    }

    private renderValue(value: unknown, indent: number, collapsed: boolean): string {
        if (value === null) {
            return '<span class="json-null">null</span>';
        }

        if (typeof value === 'boolean') {
            return `<span class="json-boolean">${value}</span>`;
        }

        if (typeof value === 'number') {
            return `<span class="json-number">${value}</span>`;
        }

        if (typeof value === 'string') {
            return `<span class="json-string">"${escapeHtml(value)}"</span>`;
        }

        if (Array.isArray(value)) {
            return this.renderArray(value, indent, collapsed);
        }

        if (typeof value === 'object') {
            return this.renderObject(value as Record<string, unknown>, indent, collapsed);
        }

        return String(value);
    }

    private renderArray(arr: unknown[], indent: number, collapsed: boolean): string {
        if (arr.length === 0) {
            return '[]';
        }

        const spaces = '  '.repeat(indent);
        const innerSpaces = '  '.repeat(indent + 1);

        const items = arr.map(item =>
            `${innerSpaces}${this.renderValue(item, indent + 1, collapsed)}`
        ).join(',\n');

        if (collapsed && arr.length > 3) {
            return `[<span class="collapsible">...</span><span class="collapsible-content">\n${items}\n${spaces}</span>]`;
        }

        return `[\n${items}\n${spaces}]`;
    }

    private renderObject(obj: Record<string, unknown>, indent: number, collapsed: boolean): string {
        const keys = Object.keys(obj);
        if (keys.length === 0) {
            return '{}';
        }

        const spaces = '  '.repeat(indent);
        const innerSpaces = '  '.repeat(indent + 1);

        const items = keys.map(key =>
            `${innerSpaces}<span class="json-key">"${escapeHtml(key)}"</span>: ${this.renderValue(obj[key], indent + 1, collapsed)}`
        ).join(',\n');

        if (collapsed && keys.length > 5) {
            return `{<span class="collapsible">...</span><span class="collapsible-content">\n${items}\n${spaces}</span>}`;
        }

        return `{\n${items}\n${spaces}}`;
    }
}
