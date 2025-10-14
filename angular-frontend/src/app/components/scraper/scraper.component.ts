import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { ReactiveFormsModule, FormBuilder, type FormGroup, Validators } from "@angular/forms"
import { MatTabsModule } from "@angular/material/tabs"
import { MatCardModule } from "@angular/material/card"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatFormFieldModule } from "@angular/material/form-field"
import { MatInputModule } from "@angular/material/input"
import { MatSelectModule } from "@angular/material/select"
import { MatCheckboxModule } from "@angular/material/checkbox"
import { MatProgressSpinnerModule } from "@angular/material/progress-spinner"
import { MatExpansionModule } from "@angular/material/expansion"
import { MatListModule } from "@angular/material/list"
import { ApiService, ScrapeRequest, BulkScrapeRequest } from "../../services/api.service"

@Component({
  selector: "app-scraper",
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatTabsModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatCheckboxModule,
    MatProgressSpinnerModule,
    MatExpansionModule,
    MatListModule,
  ],
  template: `
    <div class="scraper-container">
      <h1>Web Scraper</h1>
      
      <mat-tab-group>
        <mat-tab label="Single URL">
          <div class="tab-content">
            <form [formGroup]="singleUrlForm" (ngSubmit)="scrapeSingleUrl()">
              <mat-card>
                <mat-card-header>
                  <mat-card-title>Scrape Single URL</mat-card-title>
                </mat-card-header>
                <mat-card-content>
                  <div class="form-row">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>URL to Scrape</mat-label>
                      <input matInput formControlName="url" placeholder="https://example.com">
                      <mat-error *ngIf="singleUrlForm.get('url')?.hasError('required')">
                        URL is required
                      </mat-error>
                      <mat-error *ngIf="singleUrlForm.get('url')?.hasError('pattern')">
                        Please enter a valid URL
                      </mat-error>
                    </mat-form-field>
                  </div>

                  <div class="form-row">
                    <mat-form-field appearance="outline">
                      <mat-label>Output Format</mat-label>
                      <mat-select formControlName="output_format">
                        <mat-option value="json">JSON</mat-option>
                        <mat-option value="csv">CSV</mat-option>
                        <mat-option value="txt">Text</mat-option>
                        <mat-option value="pdf">PDF</mat-option>
                      </mat-select>
                    </mat-form-field>
                  </div>

                  <div class="form-row">
                    <h3>Extraction Options</h3>
                    <div class="checkbox-group">
                      <mat-checkbox formControlName="extract_text">Extract Text</mat-checkbox>
                      <mat-checkbox formControlName="extract_links">Extract Links</mat-checkbox>
                      <mat-checkbox formControlName="extract_images">Extract Images</mat-checkbox>
                      <mat-checkbox formControlName="extract_tables">Extract Tables</mat-checkbox>
                      <mat-checkbox formControlName="scroll_page">Scroll Page (for dynamic content)</mat-checkbox>
                    </div>
                  </div>

                  <div class="form-row">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Wait for Element (CSS Selector)</mat-label>
                      <input matInput formControlName="wait_for_element" placeholder=".content-loaded">
                      <mat-hint>CSS selector to wait for before scraping (optional)</mat-hint>
                    </mat-form-field>
                  </div>
                </mat-card-content>
                <mat-card-actions>
                  <button mat-raised-button color="primary" type="submit" [disabled]="singleUrlForm.invalid || singleUrlLoading">
                    <mat-icon *ngIf="singleUrlLoading">hourglass_empty</mat-icon>
                    <mat-icon *ngIf="!singleUrlLoading">web</mat-icon>
                    {{ singleUrlLoading ? 'Scraping...' : 'Scrape URL' }}
                  </button>
                </mat-card-actions>
              </mat-card>
            </form>

            <mat-card *ngIf="singleUrlResult" class="result-card">
              <mat-card-header>
                <mat-card-title>Scraping Result</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="result-info">
                  <p><strong>URL:</strong> {{ singleUrlResult.url }}</p>
                  <p><strong>Method Used:</strong> {{ singleUrlResult.method_used }}</p>
                  <p><strong>Status:</strong> {{ singleUrlResult.status }}</p>
                </div>
                <mat-accordion>
                  <mat-expansion-panel>
                    <mat-expansion-panel-header>
                      <mat-panel-title>Raw Content</mat-panel-title>
                    </mat-expansion-panel-header>
                    <pre>{{ singleUrlResult.content | json }}</pre>
                  </mat-expansion-panel>
                  <mat-expansion-panel>
                    <mat-expansion-panel-header>
                      <mat-panel-title>Formatted Output</mat-panel-title>
                    </mat-expansion-panel-header>
                    <pre>{{ singleUrlResult.formatted_content }}</pre>
                  </mat-expansion-panel>
                </mat-accordion>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>

        <mat-tab label="Bulk Scraping">
          <div class="tab-content">
            <form [formGroup]="bulkScrapeForm" (ngSubmit)="bulkScrape()">
              <mat-card>
                <mat-card-header>
                  <mat-card-title>Bulk URL Discovery & Scraping</mat-card-title>
                </mat-card-header>
                <mat-card-content>
                  <div class="form-row">
                    <mat-form-field appearance="outline" class="full-width">
                      <mat-label>Base URL</mat-label>
                      <input matInput formControlName="base_url" placeholder="https://example.com">
                      <mat-hint>Starting URL for discovery</mat-hint>
                    </mat-form-field>
                  </div>

                  <div class="form-row">
                    <mat-form-field appearance="outline">
                      <mat-label>Max Crawl Depth</mat-label>
                      <input matInput type="number" formControlName="max_depth" min="1" max="5">
                    </mat-form-field>

                    <mat-form-field appearance="outline">
                      <mat-label>Max URLs</mat-label>
                      <input matInput type="number" formControlName="max_urls" min="1" max="1000">
                    </mat-form-field>

                    <mat-form-field appearance="outline">
                      <mat-label>Output Format</mat-label>
                      <mat-select formControlName="output_format">
                        <mat-option value="json">JSON</mat-option>
                        <mat-option value="csv">CSV</mat-option>
                        <mat-option value="txt">Text</mat-option>
                      </mat-select>
                    </mat-form-field>
                  </div>

                  <div class="form-row">
                    <mat-checkbox formControlName="store_in_rag">Store results in RAG system</mat-checkbox>
                  </div>
                </mat-card-content>
                <mat-card-actions>
                  <button mat-raised-button color="primary" type="submit" [disabled]="bulkScrapeForm.invalid || bulkScrapeLoading">
                    <mat-icon *ngIf="bulkScrapeLoading">hourglass_empty</mat-icon>
                    <mat-icon *ngIf="!bulkScrapeLoading">language</mat-icon>
                    {{ bulkScrapeLoading ? 'Processing...' : 'Start Bulk Scraping' }}
                  </button>
                  <button mat-button (click)="discoverUrls()" [disabled]="!bulkScrapeForm.get('base_url')?.value">
                    <mat-icon>search</mat-icon>
                    Discover URLs Only
                  </button>
                </mat-card-actions>
              </mat-card>
            </form>

            <mat-card *ngIf="bulkScrapeResult" class="result-card">
              <mat-card-header>
                <mat-card-title>Bulk Scraping Status</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="result-info">
                  <p><strong>Status:</strong> {{ bulkScrapeResult.status }}</p>
                  <p><strong>URLs Discovered:</strong> {{ bulkScrapeResult.discovered_urls_count }}</p>
                  <p><strong>URLs to Scrape:</strong> {{ bulkScrapeResult.urls_to_scrape_count }}</p>
                </div>
                <mat-accordion *ngIf="bulkScrapeResult.urls_preview">
                  <mat-expansion-panel>
                    <mat-expansion-panel-header>
                      <mat-panel-title>URL Preview (First 10)</mat-panel-title>
                    </mat-expansion-panel-header>
                    <mat-list>
                      <mat-list-item *ngFor="let url of bulkScrapeResult.urls_preview">
                        <a [href]="url" target="_blank">{{ url }}</a>
                      </mat-list-item>
                    </mat-list>
                  </mat-expansion-panel>
                </mat-accordion>
              </mat-card-content>
            </mat-card>

            <mat-card *ngIf="discoveredUrls" class="result-card">
              <mat-card-header>
                <mat-card-title>Discovered URLs</mat-card-title>
              </mat-card-header>
              <mat-card-content>
                <div class="result-info">
                  <p><strong>Base URL:</strong> {{ discoveredUrls.base_url }}</p>
                  <p><strong>Total Discovered:</strong> {{ discoveredUrls.total_discovered }}</p>
                  <p><strong>Showing:</strong> {{ discoveredUrls.returned_count }}</p>
                </div>
                <mat-list class="url-list">
                  <mat-list-item *ngFor="let url of discoveredUrls.urls">
                    <a [href]="url" target="_blank">{{ url }}</a>
                  </mat-list-item>
                </mat-list>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>

        <mat-tab label="Status">
          <div class="tab-content">
            <mat-card>
              <mat-card-header>
                <mat-card-title>Scraping Status</mat-card-title>
                <button mat-icon-button (click)="refreshStatus()">
                  <mat-icon>refresh</mat-icon>
                </button>
              </mat-card-header>
              <mat-card-content>
                <div *ngIf="scrapingStatus" class="status-info">
                  <p><strong>Status:</strong> {{ scrapingStatus.status }}</p>
                  <p><strong>Documents Stored:</strong> {{ scrapingStatus.documents_stored }}</p>
                  <p><strong>Collection:</strong> {{ scrapingStatus.collection_name }}</p>
                </div>
                <div *ngIf="!scrapingStatus" class="loading">
                  <mat-spinner diameter="50"></mat-spinner>
                  <p>Loading status...</p>
                </div>
              </mat-card-content>
            </mat-card>
          </div>
        </mat-tab>
      </mat-tab-group>
    </div>
  `,
  styles: [
    `
    .scraper-container {
      padding: 20px;
    }

    .tab-content {
      padding: 20px 0;
    }

    .form-row {
      display: flex;
      gap: 20px;
      margin-bottom: 20px;
      align-items: flex-start;
    }

    .full-width {
      width: 100%;
    }

    .checkbox-group {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .result-card {
      margin-top: 20px;
    }

    .result-info {
      margin-bottom: 20px;
    }

    .result-info p {
      margin: 5px 0;
    }

    .url-list {
      max-height: 400px;
      overflow-y: auto;
    }

    .status-info p {
      margin: 10px 0;
      font-size: 1.1em;
    }

    .loading {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 20px;
    }

    pre {
      background-color: #f5f5f5;
      padding: 15px;
      border-radius: 5px;
      overflow-x: auto;
      max-height: 300px;
    }
  `,
  ],
})
export class ScraperComponent implements OnInit {
  singleUrlForm: FormGroup
  bulkScrapeForm: FormGroup
  singleUrlLoading = false
  bulkScrapeLoading = false
  singleUrlResult: any = null
  bulkScrapeResult: any = null
  discoveredUrls: any = null
  scrapingStatus: any = null

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService,
  ) {
    this.singleUrlForm = this.fb.group({
      url: ["", [Validators.required, Validators.pattern(/^https?:\/\/.+/)]],
      extract_text: [true],
      extract_links: [false],
      extract_images: [false],
      extract_tables: [false],
      output_format: ["json"],
      wait_for_element: [""],
      scroll_page: [false],
    })

    this.bulkScrapeForm = this.fb.group({
      base_url: ["", [Validators.required, Validators.pattern(/^https?:\/\/.+/)]],
      max_depth: [2, [Validators.required, Validators.min(1), Validators.max(5)]],
      max_urls: [100, [Validators.required, Validators.min(1), Validators.max(1000)]],
      output_format: ["json"],
      store_in_rag: [true],
    })
  }

  ngOnInit() {
    this.refreshStatus()
  }

  scrapeSingleUrl() {
    if (this.singleUrlForm.valid) {
      this.singleUrlLoading = true
      this.singleUrlResult = null

      const request: ScrapeRequest = this.singleUrlForm.value

      this.apiService.scrapeSingleUrl(request).subscribe({
        next: (result) => {
          this.singleUrlResult = result
          this.singleUrlLoading = false
        },
        error: (error) => {
          console.error("Scraping error:", error)
          alert("Error scraping URL: " + (error.error?.detail || error.message))
          this.singleUrlLoading = false
        },
      })
    }
  }

  bulkScrape() {
    if (this.bulkScrapeForm.valid) {
      this.bulkScrapeLoading = true
      this.bulkScrapeResult = null

      const request: BulkScrapeRequest = {
        ...this.bulkScrapeForm.value,
        scrape_params: {
          extract_text: true,
          extract_links: false,
          extract_images: false,
          extract_tables: false,
        },
      }

      this.apiService.bulkScrape(request).subscribe({
        next: (result) => {
          this.bulkScrapeResult = result
          this.bulkScrapeLoading = false
        },
        error: (error) => {
          console.error("Bulk scraping error:", error)
          alert("Error starting bulk scraping: " + (error.error?.detail || error.message))
          this.bulkScrapeLoading = false
        },
      })
    }
  }

  discoverUrls() {
    const baseUrl = this.bulkScrapeForm.get("base_url")?.value
    const maxDepth = this.bulkScrapeForm.get("max_depth")?.value || 2
    const maxUrls = this.bulkScrapeForm.get("max_urls")?.value || 100

    if (baseUrl) {
      this.apiService.discoverUrls(baseUrl, maxDepth, maxUrls).subscribe({
        next: (result) => {
          this.discoveredUrls = result
        },
        error: (error) => {
          console.error("URL discovery error:", error)
          alert("Error discovering URLs: " + (error.error?.detail || error.message))
        },
      })
    }
  }

  refreshStatus() {
    this.apiService.getScrapingStatus().subscribe({
      next: (status) => {
        this.scrapingStatus = status
      },
      error: (error) => {
        console.error("Error loading status:", error)
      },
    })
  }
}
