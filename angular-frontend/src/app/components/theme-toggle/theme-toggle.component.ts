import { Component, type OnInit } from "@angular/core"
import { CommonModule } from "@angular/common"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatMenuModule } from "@angular/material/menu"
import { ThemeService, Theme } from "../../services/theme.service"

@Component({
  selector: "app-theme-toggle",
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatIconModule, MatMenuModule],
  template: `
    <button mat-icon-button [matMenuTriggerFor]="themeMenu" aria-label="Theme selector">
      <mat-icon>{{ getThemeIcon() }}</mat-icon>
    </button>
    
    <mat-menu #themeMenu="matMenu">
      <button mat-menu-item (click)="setTheme('light')">
        <mat-icon>light_mode</mat-icon>
        <span>Light</span>
      </button>
      
      <button mat-menu-item (click)="setTheme('dark')">
        <mat-icon>dark_mode</mat-icon>
        <span>Dark</span>
      </button>
      
      <button mat-menu-item (click)="setTheme('system')">
        <mat-icon>devices</mat-icon>
        <span>System</span>
      </button>
    </mat-menu>
  `,
  styles: [
    `
    button {
      min-width: 0;
    }
  `,
  ],
})
export class ThemeToggleComponent implements OnInit {
  currentTheme: Theme = "system"

  constructor(private themeService: ThemeService) {}

  ngOnInit(): void {
    this.themeService.theme$.subscribe((theme) => {
      this.currentTheme = theme
    })
  }

  setTheme(theme: Theme): void {
    this.themeService.setTheme(theme)
  }

  getThemeIcon(): string {
    switch (this.currentTheme) {
      case "light":
        return "light_mode"
      case "dark":
        return "dark_mode"
      case "system":
        return "devices"
      default:
        return "brightness_auto"
    }
  }
}
