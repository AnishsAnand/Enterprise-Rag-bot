// register.component.ts - COMPLETE FIXED VERSION
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

@Component({
  selector: 'app-register',
  standalone: true,
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css'],
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
  ],
})
export class RegisterComponent {
  username = '';
  email = '';
  password = '';
  confirmPassword = '';
  loading = false;
  errorMessage: string | null = null;
  hidePassword = true;
  hideConfirmPassword = true;

  constructor(
    private authService: AuthService,
    private router: Router,
    private snackBar: MatSnackBar
  ) {}

  async register() {
    // Reset error
    this.errorMessage = null;

    // Validation
    if (!this.username || !this.email || !this.password || !this.confirmPassword) {
      this.errorMessage = 'Please fill in all fields';
      this.showSnackBar('Please fill in all fields', 'error');
      return;
    }

    // Username validation
    if (this.username.length < 3) {
      this.errorMessage = 'Username must be at least 3 characters';
      this.showSnackBar('Username must be at least 3 characters', 'error');
      return;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(this.email)) {
      this.errorMessage = 'Please enter a valid email address';
      this.showSnackBar('Please enter a valid email address', 'error');
      return;
    }

    // Password validation
    if (this.password.length < 6) {
      this.errorMessage = 'Password must be at least 6 characters';
      this.showSnackBar('Password must be at least 6 characters', 'error');
      return;
    }

    // Password confirmation
    if (this.password !== this.confirmPassword) {
      this.errorMessage = 'Passwords do not match';
      this.showSnackBar('Passwords do not match', 'error');
      return;
    }

    this.loading = true;

    try {
      // Call auth service register
      await this.authService.register({
        username: this.username,
        email: this.email,
        password: this.password,
      }).toPromise();
      
      // Success
      this.showSnackBar('Registration successful! Redirecting...', 'success');
      
      // Navigate based on user role
      setTimeout(() => {
        const currentUser = this.authService.getCurrentUserValue();
        if (currentUser?.role === 'admin') {
          this.router.navigate(['/admin']);
        } else {
          this.router.navigate(['/user']);
        }
      }, 1500);
      
    } catch (error: any) {
      console.error('Registration error:', error);
      
      let errorMsg = 'Registration failed. Please try again.';
      
      if (error?.status === 400) {
        errorMsg = 'User already exists';
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

  goToLogin() {
    this.router.navigate(['/login']);
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
      this.register();
    }
  }
}