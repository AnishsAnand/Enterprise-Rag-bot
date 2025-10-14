import { Component } from '@angular/core';
import { RagWidgetComponent } from './rag-widget/rag-widget.component';

@Component({
  selector: 'app-widget-embed',
  standalone: true,
  imports: [RagWidgetComponent],
  templateUrl: './widget-embed.component.html',
  styleUrls: ['./widget-embed.component.scss']
})
export class WidgetEmbedComponent {}
