// src/app/app.routes.ts
import { Routes } from '@angular/router';
import { ChatComponent } from './components/chat/chat.component';
import { WidgetEmbedComponent } from './components/widget-embed/widget-embed.component';

export const appRoutes: Routes = [
  {
    path: '',
    component: ChatComponent,   // main chat UI
  },
  {
    path: 'widget-embed',
    component: WidgetEmbedComponent,  // iframe version for embedding
  },
  {
    path: '**',
    redirectTo: '',   // fallback to chat
  }
];
