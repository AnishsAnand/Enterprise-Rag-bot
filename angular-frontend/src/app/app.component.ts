import { Component, OnInit, Renderer2 } from '@angular/core';
import { RouterOutlet, Router, NavigationEnd } from '@angular/router';
import { CommonModule } from '@angular/common';
import { RagWidgetComponent } from './components/rag-widget/rag-widget.component';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RagWidgetComponent],
  templateUrl: './app.component.html',
})
export class AppComponent implements OnInit {
  isPublicEmbed = false;

  constructor(private router: Router, private renderer: Renderer2) {
    const urlParams = new URLSearchParams(window.location.search);
    this.isPublicEmbed = urlParams.get('public') === 'true';
  }

  ngOnInit(): void {
    this.router.events.pipe(
      filter(event => event instanceof NavigationEnd)
    ).subscribe((event: any) => {
      // Clear all possible body classes
      this.renderer.removeClass(document.body, 'login-page');
      this.renderer.removeClass(document.body, 'dashboard-page');
      this.renderer.removeClass(document.body, 'widget-page');

      const url = event.urlAfterRedirects;

      if (this.isPublicEmbed || url.startsWith('/widget-embed')) {
        this.renderer.addClass(document.body, 'widget-page');
      } else if (url.startsWith('/login')) {
        this.renderer.addClass(document.body, 'login-page');
      } else {
        this.renderer.addClass(document.body, 'dashboard-page');
      }
    });
  }
}
