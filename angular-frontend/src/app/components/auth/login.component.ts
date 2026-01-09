// login.component.ts - COMPLETE FIXED VERSION
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';

import { AuthService } from '../../services/auth.service';
import { RagWidgetComponent } from '../rag-widget/rag-widget.component';

@Component({
  selector: 'app-login',
  standalone: true,
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatIconModule,
    MatSnackBarModule,
    RagWidgetComponent
  ],
})
export class LoginComponent {
  username = '';
  password = '';
  loading = false;
  errorMessage: string | null = null;
  hidePassword = true;

  constructor(
    private authService: AuthService,
    private router: Router,
    private snackBar: MatSnackBar
  ) {}

  async login() {
    // Validation
    if (!this.username || !this.password) {
      this.errorMessage = 'Please enter both username and password';
      this.showSnackBar('Please enter both username and password', 'error');
      return;
    }

    this.loading = true;
    this.errorMessage = null;

    try {
      // Call auth service login
      await this.authService.login(this.username, this.password).toPromise();
      
      // Success - show message and navigate
      this.showSnackBar('Login successful!', 'success');
      
      // Check user role and navigate accordingly
      const currentUser = this.authService.getCurrentUserValue();
      if (currentUser?.role === 'admin') {
        this.router.navigate(['/admin']);
      } else {
        this.router.navigate(['/user']);
      }
      
    } catch (error: any) {
      console.error('Login error:', error);
      
      // Handle different error types
      let errorMsg = 'Login failed. Please try again.';
      
      if (error?.status === 401) {
        errorMsg = 'Invalid username or password';
      } else if (error?.status === 403) {
        errorMsg = 'Account disabled. Please contact support.';
      } else if (error?.status === 0) {
        errorMsg = 'Cannot connect to server. Please check your connection.';
      } else if (error?.error?.detail) {
        errorMsg = error.error.detail;
      } else if (error?.message) {
        errorMsg = error.message;
      }
      
      this.errorMessage = errorMsg;
      this.showSnackBar(errorMsg, 'error');
      
    } finally {
      this.loading = false;
    }
  }

  goToRegister() {
    this.router.navigate(['/register']);
  }

  private showSnackBar(message: string, type: 'success' | 'error') {
    this.snackBar.open(message, 'Close', {
      duration: type === 'success' ? 3000 : 5000,
      horizontalPosition: 'center',
      verticalPosition: 'top',
      panelClass: type === 'success' ? 'snackbar-success' : 'snackbar-error'
    });
  }

  onKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !this.loading) {
      this.login();
    }
  }
}