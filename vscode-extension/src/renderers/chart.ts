import { ArtifactRenderer, ChartArtifact, Artifact } from '../types';
import { escapeHtml } from './base';

/**
 * Chart renderer using simple SVG charts
 */
export class ChartRenderer implements ArtifactRenderer {
    type = 'chart' as const;

    render(artifact: Artifact): string {
        const chart = artifact as ChartArtifact;
        const { chartType, labels, datasets } = chart.data;

        switch (chartType) {
            case 'bar':
                return this.renderBarChart(labels, datasets, chart.title);
            case 'line':
                return this.renderLineChart(labels, datasets, chart.title);
            case 'pie':
                return this.renderPieChart(labels, datasets[0], chart.title);
            default:
                return this.renderBarChart(labels, datasets, chart.title);
        }
    }

    private renderBarChart(
        labels: string[],
        datasets: ChartArtifact['data']['datasets'],
        title?: string
    ): string {
        const width = 600;
        const height = 300;
        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;

        const allValues = datasets.flatMap(d => d.data);
        const maxValue = Math.max(...allValues, 0);
        const barWidth = chartWidth / labels.length / datasets.length - 4;

        const colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'];

        let bars = '';
        datasets.forEach((dataset, di) => {
            const color = dataset.color || colors[di % colors.length];
            dataset.data.forEach((value, i) => {
                const x = padding + i * (chartWidth / labels.length) + di * (barWidth + 2);
                const barHeight = (value / maxValue) * chartHeight;
                const y = padding + chartHeight - barHeight;
                bars += `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${color}" opacity="0.8">
                    <title>${escapeHtml(dataset.label)}: ${value}</title>
                </rect>`;
            });
        });

        const xLabels = labels.map((label, i) => {
            const x = padding + i * (chartWidth / labels.length) + (chartWidth / labels.length) / 2;
            return `<text x="${x}" y="${height - 10}" text-anchor="middle" fill="currentColor" font-size="12">${escapeHtml(label)}</text>`;
        }).join('');

        const legend = datasets.map((dataset, i) => {
            const color = dataset.color || colors[i % colors.length];
            return `<span style="margin-right: 16px;"><span style="display: inline-block; width: 12px; height: 12px; background: ${color}; margin-right: 4px;"></span>${escapeHtml(dataset.label)}</span>`;
        }).join('');

        return `
            <div class="chart-container">
                ${title ? `<h3>${escapeHtml(title)}</h3>` : ''}
                <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
                    ${bars}
                    ${xLabels}
                </svg>
                <div style="text-align: center; margin-top: 8px;">${legend}</div>
            </div>
        `;
    }

    private renderLineChart(
        labels: string[],
        datasets: ChartArtifact['data']['datasets'],
        title?: string
    ): string {
        const width = 600;
        const height = 300;
        const padding = 40;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;

        const allValues = datasets.flatMap(d => d.data);
        const maxValue = Math.max(...allValues, 0);

        const colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0'];

        let lines = '';
        datasets.forEach((dataset, di) => {
            const color = dataset.color || colors[di % colors.length];
            const points = dataset.data.map((value, i) => {
                const x = padding + i * (chartWidth / (labels.length - 1));
                const y = padding + chartHeight - (value / maxValue) * chartHeight;
                return `${x},${y}`;
            }).join(' ');
            lines += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2"/>`;

            // Add dots
            dataset.data.forEach((value, i) => {
                const x = padding + i * (chartWidth / (labels.length - 1));
                const y = padding + chartHeight - (value / maxValue) * chartHeight;
                lines += `<circle cx="${x}" cy="${y}" r="4" fill="${color}">
                    <title>${escapeHtml(dataset.label)}: ${value}</title>
                </circle>`;
            });
        });

        const xLabels = labels.map((label, i) => {
            const x = padding + i * (chartWidth / (labels.length - 1));
            return `<text x="${x}" y="${height - 10}" text-anchor="middle" fill="currentColor" font-size="12">${escapeHtml(label)}</text>`;
        }).join('');

        const legend = datasets.map((dataset, i) => {
            const color = dataset.color || colors[i % colors.length];
            return `<span style="margin-right: 16px;"><span style="display: inline-block; width: 12px; height: 12px; background: ${color}; margin-right: 4px;"></span>${escapeHtml(dataset.label)}</span>`;
        }).join('');

        return `
            <div class="chart-container">
                ${title ? `<h3>${escapeHtml(title)}</h3>` : ''}
                <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
                    ${lines}
                    ${xLabels}
                </svg>
                <div style="text-align: center; margin-top: 8px;">${legend}</div>
            </div>
        `;
    }

    private renderPieChart(
        labels: string[],
        dataset: ChartArtifact['data']['datasets'][0],
        title?: string
    ): string {
        const size = 300;
        const cx = size / 2;
        const cy = size / 2;
        const radius = size / 2 - 20;

        const total = dataset.data.reduce((a, b) => a + b, 0);
        const colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4', '#795548'];

        let currentAngle = -90;
        const slices = dataset.data.map((value, i) => {
            const angle = (value / total) * 360;
            const startAngle = currentAngle;
            const endAngle = currentAngle + angle;
            currentAngle = endAngle;

            const startRad = (startAngle * Math.PI) / 180;
            const endRad = (endAngle * Math.PI) / 180;

            const x1 = cx + radius * Math.cos(startRad);
            const y1 = cy + radius * Math.sin(startRad);
            const x2 = cx + radius * Math.cos(endRad);
            const y2 = cy + radius * Math.sin(endRad);

            const largeArc = angle > 180 ? 1 : 0;
            const color = colors[i % colors.length];

            return `<path d="M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z" fill="${color}" opacity="0.8">
                <title>${escapeHtml(labels[i])}: ${value} (${((value / total) * 100).toFixed(1)}%)</title>
            </path>`;
        }).join('');

        const legend = labels.map((label, i) => {
            const color = colors[i % colors.length];
            const percent = ((dataset.data[i] / total) * 100).toFixed(1);
            return `<div style="margin: 4px 0;"><span style="display: inline-block; width: 12px; height: 12px; background: ${color}; margin-right: 8px;"></span>${escapeHtml(label)} (${percent}%)</div>`;
        }).join('');

        return `
            <div class="chart-container" style="display: flex; align-items: center; gap: 32px;">
                <div>
                    ${title ? `<h3>${escapeHtml(title)}</h3>` : ''}
                    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
                        ${slices}
                    </svg>
                </div>
                <div>${legend}</div>
            </div>
        `;
    }
}
