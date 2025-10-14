import { Routes } from '@angular/router';
import { LoginComponent } from './components/auth/login.component';
import { RegisterComponent } from './components/auth/register.component';
import { AdminComponent } from './components/admin/admin.component';
import { LayoutComponent } from './layout/layout.component';
import { AuthGuard } from './guards/auth.guard';
import { RagWidgetComponent } from './components/rag-widget/rag-widget.component';
import { UserDashboardComponent } from './user-dashboard.component';
import { WidgetEmbedComponent } from './components/widget-embed.component';

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  {
    path: 'embed',
    component: WidgetEmbedComponent,
  },

  {
    path: '',
    component: LayoutComponent,
    canActivate: [AuthGuard],
    children: [
      { path: 'admin', component: AdminComponent },
      { path: 'rag_widget', component: RagWidgetComponent },
      { path: 'user', component: UserDashboardComponent },
    ],
  },

  { path: '**', redirectTo: 'login' },
];
