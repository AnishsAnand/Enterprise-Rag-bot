import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { RouterModule } from "@angular/router"
import { MatCardModule } from "@angular/material/card"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { MatChipsModule } from "@angular/material/chips"
import { ApiService } from "../../services/api.service"

@Component({
  selector: "app-dashboard",
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatChipsModule,
  ],
  template: `
    <div class="dashboard-container">
      <h1>System Dashboard</h1>
      
      <div class="stats-grid">
        <mat-card class="stat-card">
          <mat-card-header>
            <mat-card-title>
              <mat-icon>web</mat-icon>
              Scraping Status
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div *ngIf="scrapingStats" class="stat-content">
              <div class="stat-number">{{ scrapingStats.documents_stored || 0 }}</div>
              <div class="stat-label">Documents Scraped</div>
              <mat-chip-set>
                <mat-chip [color]="scrapingStats.status === 'active' ? 'primary' : 'accent'">
                  {{ scrapingStats.status || 'idle' }}
                </mat-chip>
              </mat-chip-set>
            </div>
            <div *ngIf="!scrapingStats" class="loading">
              <mat-spinner diameter="40"></mat-spinner>
            </div>
          </mat-card-content>
          <mat-card-actions>
            <button mat-button (click)="refreshStats()">
              <mat-icon>refresh</mat-icon>
              Refresh
            </button>
          </mat-card-actions>
        </mat-card>

        <mat-card class="stat-card">
          <mat-card-header>
            <mat-card-title>
              <mat-icon>psychology</mat-icon>
              RAG System
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div *ngIf="ragStats" class="stat-content">
              <div class="stat-number">{{ ragStats.document_count || 0 }}</div>
              <div class="stat-label">Documents in RAG</div>
              <mat-chip-set>
                <mat-chip color="primary">
                  {{ ragStats.collection_name || 'default' }}
                </mat-chip>
              </mat-chip-set>
            </div>
            <div *ngIf="!ragStats" class="loading">
              <mat-spinner diameter="40"></mat-spinner>
            </div>
          </mat-card-content>
          <mat-card-actions>
            <button mat-button (click)="refreshStats()">
              <mat-icon>refresh</mat-icon>
              Refresh
            </button>
          </mat-card-actions>
        </mat-card>

        <mat-card class="stat-card">
          <mat-card-header>
            <mat-card-title>
              <mat-icon>health_and_safety</mat-icon>
              System Health
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div *ngIf="systemHealth" class="stat-content">
              <div class="health-indicators">
                <div class="health-item">
                  <mat-icon [color]="systemHealth.database === 'healthy' ? 'primary' : 'warn'">
                    {{ systemHealth.database === 'healthy' ? 'check_circle' : 'error' }}
                  </mat-icon>
                  <span>Database</span>
                </div>
                <div class="health-item">
                  <mat-icon [color]="systemHealth.ai_service === 'healthy' ? 'primary' : 'warn'">
                    {{ systemHealth.ai_service === 'healthy' ? 'check_circle' : 'error' }}
                  </mat-icon>
                  <span>AI Service</span>
                </div>
                <div class="health-item">
                  <mat-icon [color]="systemHealth.vector_db === 'healthy' ? 'primary' : 'warn'">
                    {{ systemHealth.vector_db === 'healthy' ? 'check_circle' : 'error' }}
                  </mat-icon>
                  <span>Vector DB</span>
                </div>
              </div>
            </div>
            <div *ngIf="!systemHealth" class="loading">
              <mat-spinner diameter="40"></mat-spinner>
            </div>
          </mat-card-content>
          <mat-card-actions>
            <button mat-button (click)="refreshStats()">
              <mat-icon>refresh</mat-icon>
              Refresh
            </button>
          </mat-card-actions>
        </mat-card>

        <mat-card class="stat-card">
          <mat-card-header>
            <mat-card-title>
              <mat-icon>analytics</mat-icon>
              System Stats
            </mat-card-title>
          </mat-card-header>
          <mat-card-content>
            <div *ngIf="systemStats" class="stat-content">
              <div class="stats-list">
                <div class="stat-item">
                  <span class="stat-value">{{ systemStats.uptime || '0h' }}</span>
                  <span class="stat-name">Uptime</span>
                </div>
                <div class="stat-item">
                  <span class="stat-value">{{ systemStats.memory_usage || '0%' }}</span>
                  <span class="stat-name">Memory</span>
                </div>
                <div class="stat-item">
                  <span class="stat-value">{{ systemStats.cpu_usage || '0%' }}</span>
                  <span class="stat-name">CPU</span>
                </div>
              </div>
            </div>
            <div *ngIf="!systemStats" class="loading">
              <mat-spinner diameter="40"></mat-spinner>
            </div>
          </mat-card-content>
          <mat-card-actions>
            <button mat-button (click)="refreshStats()">
              <mat-icon>refresh</mat-icon>
              Refresh
            </button>
          </mat-card-actions>
        </mat-card>
      </div>

      <div class="quick-actions">
        <h2>Quick Actions</h2>
        <div class="actions-grid">
          <button mat-raised-button color="primary" routerLink="/scraper">
            <mat-icon>web</mat-icon>
            Start Scraping
          </button>
          <button mat-raised-button color="accent" routerLink="/rag">
            <mat-icon>psychology</mat-icon>
            Query RAG
          </button>
          <button mat-raised-button routerLink="/admin">
            <mat-icon>settings</mat-icon>
            System Settings
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [
    `
    .dashboard-container {
      padding: 20px;
      max-width: 1200px;
      margin: 0 auto;
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }

    .stat-card {
      min-height: 200px;
    }

    .stat-content {
      text-align: center;
    }

    .stat-number {
      font-size: 3em;
      font-weight: bold;
      color: #1976d2;
      margin-bottom: 10px;
    }

    .stat-label {
      font-size: 1.1em;
      color: #666;
      margin-bottom: 15px;
    }

    .loading {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100px;
    }

    .health-indicators {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .health-item {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .stats-list {
      display: flex;
      flex-direction: column;
      gap: 15px;
    }

    .stat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .stat-value {
      font-weight: bold;
      color: #1976d2;
    }

    .stat-name {
      color: #666;
    }

    .quick-actions {
      margin-top: 40px;
    }

    .actions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin-top: 20px;
    }

    .actions-grid button {
      height: 60px;
      font-size: 1.1em;
    }

    .actions-grid button mat-icon {
      margin-right: 8px;
    }
  `,
  ],
})
export class DashboardComponent implements OnInit {
  scrapingStats: any = null
  ragStats: any = null
  systemHealth: any = null
  systemStats: any = null

  constructor(private apiService: ApiService) {}

  ngOnInit() {
    this.refreshStats()
  }

  refreshStats() {
    // Load scraping stats
    this.apiService.getScrapingStatus().subscribe({
      next: (stats) => {
        this.scrapingStats = stats
      },
      error: (error) => {
        console.error("Error loading scraping stats:", error)
        this.scrapingStats = { status: "error", documents_stored: 0 }
      },
    })

    // Load RAG stats
    this.apiService.getRagStats().subscribe({
      next: (stats) => {
        this.ragStats = stats
      },
      error: (error) => {
        console.error("Error loading RAG stats:", error)
        this.ragStats = { document_count: 0, collection_name: "error" }
      },
    })

    // Load system health
    this.apiService.getSystemHealth().subscribe({
      next: (health) => {
        this.systemHealth = health
      },
      error: (error) => {
        console.error("Error loading system health:", error)
        this.systemHealth = {
          database: "error",
          ai_service: "error",
          vector_db: "error",
        }
      },
    })

    // Load system stats
    this.apiService.getSystemStats().subscribe({
      next: (stats) => {
        this.systemStats = stats
      },
      error: (error) => {
        console.error("Error loading system stats:", error)
        this.systemStats = {
          uptime: "N/A",
          memory_usage: "N/A",
          cpu_usage: "N/A",
        }
      },
    })
  }
}
