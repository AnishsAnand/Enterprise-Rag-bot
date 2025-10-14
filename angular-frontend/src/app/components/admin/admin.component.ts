import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { ApiService } from '../../services/api.service';
import { Router } from '@angular/router';
import { RagWidgetComponent } from '../rag-widget/rag-widget.component'; 

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    RagWidgetComponent, 
  ],
  template: `
    <mat-card>
      <h2>Admin Dashboard</h2>
      <p><strong>Docs in RAG:</strong> {{ ragStats?.document_count || 0 }}</p>

      <button mat-raised-button color="primary" (click)="refresh()">Refresh</button>
      <button mat-raised-button color="warn" (click)="logout()">Logout</button>
    </mat-card>
    <app-rag-widget></app-rag-widget>
  `
})
export class AdminComponent implements OnInit {
  systemStatus = '';
  systemUptime = '';
  ragStats: any;

  constructor(private api: ApiService, private router: Router) {}

  ngOnInit() {
    this.refresh();
  }

  refresh() {
    this.api.getSystemHealth().subscribe(
      (res) => {
        this.systemStatus = res.status || 'Unknown';
        this.systemUptime = res.uptime || '-';
      },
      (err) => {
        this.systemStatus = 'Error';
        this.systemUptime = '-';
      }
    );

    this.api.getRagStats().subscribe(
      (stats) => {
        this.ragStats = stats;
      },
      () => {
        this.ragStats = { document_count: 0 };
      }
    );
  }

  logout() {
    localStorage.removeItem('token');
    this.router.navigate(['/login']);
  }
}
