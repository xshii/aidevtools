import { ArtifactRenderer, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Image artifact
 */
export interface ImageArtifact extends Artifact {
    type: 'image';
    data: {
        src: string;          // base64 data URL or file path
        alt?: string;
        width?: number;
        height?: number;
    };
}

/**
 * Image renderer
 */
export class ImageRenderer implements ArtifactRenderer {
    type = 'image' as const;

    render(artifact: Artifact): string {
        const img = artifact as ImageArtifact;
        const { src, alt, width, height } = img.data;

        const style = [
            'max-width: 100%',
            'height: auto',
            width ? `width: ${width}px` : '',
            height ? `height: ${height}px` : '',
        ].filter(Boolean).join('; ');

        return `
            <div style="text-align: center;">
                ${img.title ? `<h3>${escapeHtml(img.title)}</h3>` : ''}
                <img src="${escapeHtml(src)}" alt="${escapeHtml(alt || '')}" style="${style}" />
                ${alt ? `<p style="color: var(--vscode-descriptionForeground); margin-top: 8px;">${escapeHtml(alt)}</p>` : ''}
            </div>
        `;
    }
}
