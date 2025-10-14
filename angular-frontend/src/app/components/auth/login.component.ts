import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Router } from '@angular/router';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { RagWidgetComponent } from '../rag-widget/rag-widget.component';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatCardModule,
    RagWidgetComponent,
  ],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent {
  username = '';
  password = '';
  errorMessage = '';
  loading = false;

  private apiUrl = 'http://localhost:8000/api/auth/login';

  constructor(private http: HttpClient, private router: Router) {}

  async login() {
    this.errorMessage = '';
    this.loading = true;

    const body = new HttpParams()
      .set('username', this.username)
      .set('password', this.password);

    const headers = new HttpHeaders({
      'Content-Type': 'application/x-www-form-urlencoded',
    });

    try {
      const response: any = await this.http
        .post(this.apiUrl, body.toString(), { headers })
        .toPromise();

      localStorage.setItem('token', response.access_token);
      const tokenPayload = JSON.parse(atob(response.access_token.split('.')[1]));
      const userRole = tokenPayload.role;

      if (userRole === 'admin') {
        this.router.navigate(['/admin']);
      } else {
        this.router.navigate(['/user']);
      }
    } catch (err: any) {
      this.errorMessage =
        err?.error?.detail ||
        (Array.isArray(err?.error)
          ? err.error.map((e: any) => e.msg).join(', ')
          : 'Login failed.');
    } finally {
      this.loading = false;
    }
  }
  goToRegister() {
    this.router.navigate(['/register']);
  }

  logout() {
    localStorage.removeItem('token');
    this.router.navigate(['/login']);
  }
}
