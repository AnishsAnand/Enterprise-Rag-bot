// auth.guard.ts - COMPLETE FIXED VERSION
import { Injectable } from '@angular/core';
import { 
  CanActivate, 
  ActivatedRouteSnapshot, 
  RouterStateSnapshot, 
  Router,
  UrlTree 
} from '@angular/router';
import { Observable } from 'rxjs';
import { AuthService } from '../services/auth.service';

@Injectable({
  providedIn: 'root'
})
export class AuthGuard implements CanActivate {
  
  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  canActivate(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot
  ): Observable<boolean | UrlTree> | Promise<boolean | UrlTree> | boolean | UrlTree {
    
    // Check if user is authenticated
    if (this.authService.isAuthenticated()) {
      console.log('✅ Auth guard: User authenticated');
      
      // Check for role-based access if required
      const requiredRole = route.data['role'];
      
      if (requiredRole) {
        const currentUser = this.authService.getCurrentUserValue();
        
        if (currentUser?.role === requiredRole) {
          return true;
        } else {
          console.log('❌ Auth guard: Insufficient permissions');
          // Redirect to appropriate page based on user role
          if (currentUser?.role === 'admin') {
            return this.router.createUrlTree(['/admin']);
          } else {
            return this.router.createUrlTree(['/user']);
          }
        }
      }
      
      return true;
    }

    // Not authenticated - redirect to login
    console.log('❌ Auth guard: User not authenticated, redirecting to login');
    return this.router.createUrlTree(['/login'], {
      queryParams: { returnUrl: state.url }
    });
  }
}

// Additional guard for admin-only routes
@Injectable({
  providedIn: 'root'
})
export class AdminGuard implements CanActivate {
  
  constructor(
    private authService: AuthService,
    private router: Router
  ) {}

  canActivate(
    route: ActivatedRouteSnapshot,
    state: RouterStateSnapshot
  ): boolean | UrlTree {
    
    if (this.authService.isAuthenticated() && this.authService.isAdmin()) {
      console.log('✅ Admin guard: Access granted');
      return true;
    }

    console.log('❌ Admin guard: Access denied');
    
    // If authenticated but not admin, redirect to user dashboard
    if (this.authService.isAuthenticated()) {
      return this.router.createUrlTree(['/user']);
    }
    
    // Not authenticated, redirect to login
    return this.router.createUrlTree(['/login']);
  }
}