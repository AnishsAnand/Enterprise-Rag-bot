import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RagWidgetComponent } from '../components/rag-widget/rag-widget.component';
import { CommonModule } from '@angular/common';



@Component({
  selector: 'app-layout',
  standalone: true,
  imports: [CommonModule,RouterOutlet, RagWidgetComponent],
  template: `
    <router-outlet></router-outlet>
    <app-rag-widget *ngIf="isAdmin"></app-rag-widget>
  `,
})
export class LayoutComponent {
  isAdmin = false;

  ngOnInit() {
    const user = localStorage.getItem('user');
    if (user) {
      try {
        const parsed = JSON.parse(user);
        this.isAdmin = parsed?.username === 'admin';
      } catch (e) {
        console.error('Failed to parse user from localStorage:', e);
      }
    }
  }
}
