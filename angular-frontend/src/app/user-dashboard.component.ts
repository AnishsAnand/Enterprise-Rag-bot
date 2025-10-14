import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RagWidgetComponent } from './components/rag-widget/rag-widget.component';

@Component({
  selector: 'app-user-dashboard',
  standalone: true,
  imports: [CommonModule, RagWidgetComponent],
  template: `
    <h2>Welcome, user!</h2>
    <app-rag-widget></app-rag-widget>
  `,
})
export class UserDashboardComponent {}
