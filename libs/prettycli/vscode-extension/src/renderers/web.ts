import { ArtifactRenderer, WebArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Web content renderer (HTML/URL)
 */
export class WebRenderer implements ArtifactRenderer {
    type = 'web' as const;

    render(artifact: Artifact): string {
        const web = artifact as WebArtifact;
        const { html, url } = web.data;

        if (url) {
            return `
                <div>
                    ${web.title ? `<h3>${escapeHtml(web.title)}</h3>` : ''}
                    <div style="border: 1px solid var(--border-color); border-radius: 4px; overflow: hidden;">
                        <div style="padding: 8px; background: var(--header-bg); border-bottom: 1px solid var(--border-color);">
                            <a href="${escapeHtml(url)}" style="color: var(--link-color);">${escapeHtml(url)}</a>
                        </div>
                        <iframe src="${escapeHtml(url)}" sandbox="allow-scripts allow-same-origin" style="min-height: 500px;"></iframe>
                    </div>
                </div>
            `;
        }

        if (html) {
            return `
                <div>
                    ${web.title ? `<h3>${escapeHtml(web.title)}</h3>` : ''}
                    <div style="border: 1px solid var(--border-color); border-radius: 4px; padding: 16px;">
                        ${html}
                    </div>
                </div>
            `;
        }

        return '<div>No content provided</div>';
    }
}
