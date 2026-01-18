import { ArtifactRenderer, MarkdownArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Markdown renderer (basic support)
 */
export class MarkdownRenderer implements ArtifactRenderer {
    type = 'markdown' as const;

    render(artifact: Artifact): string {
        const md = artifact as MarkdownArtifact;
        const html = this.parseMarkdown(md.data.content);

        return `
            <div class="markdown-content">
                ${md.title ? `<h2>${escapeHtml(md.title)}</h2>` : ''}
                ${html}
            </div>
        `;
    }

    private parseMarkdown(content: string): string {
        let html = escapeHtml(content);

        // Headers
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Bold and italic
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Code blocks
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" style="color: var(--link-color);">$1</a>');

        // Lists
        html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // Line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = `<p>${html}</p>`;

        // Clean up empty paragraphs
        html = html.replace(/<p>\s*<\/p>/g, '');
        html = html.replace(/<p>\s*<(h[1-6]|ul|ol|pre)/g, '<$1');
        html = html.replace(/<\/(h[1-6]|ul|ol|pre)>\s*<\/p>/g, '</$1>');

        return html;
    }
}
