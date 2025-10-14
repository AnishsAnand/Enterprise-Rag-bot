import { Component, type OnInit, ViewChild, type ElementRef } from "@angular/core"
import { CommonModule } from "@angular/common"
import { FormBuilder, type FormGroup, Validators, ReactiveFormsModule } from "@angular/forms"
import { MatButtonModule } from "@angular/material/button"
import { MatIconModule } from "@angular/material/icon"
import { MatCardModule } from "@angular/material/card"
import { MatFormFieldModule } from "@angular/material/form-field"
import { MatInputModule } from "@angular/material/input"
import { MatSelectModule } from "@angular/material/select"
import { MatTabsModule } from "@angular/material/tabs"
import { MatExpansionModule } from "@angular/material/expansion"
import { ApiService } from "../../services/api.service"

interface ChatMessage {
  id: string
  message: string
  sender: "user" | "agent"
  timestamp: string
  type: "text"
}

interface SupportArticle {
  id: string
  title: string
  content: string
  category: string
  tags: string[]
  helpful_count: number
}

@Component({
  selector: "app-support-widget",
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatButtonModule,
    MatIconModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatTabsModule,
    MatExpansionModule,
  ],
  templateUrl: "./support-widget.component.html",
  styleUrls: ["./support-widget.component.css"],
})
export class SupportWidgetComponent implements OnInit {
  @ViewChild("chatMessages") chatMessages!: ElementRef
  @ViewChild("messageInput") messageInput!: ElementRef

  isExpanded = false
  chatLoading = false
  ticketLoading = false

  chatForm: FormGroup
  chatHistory: ChatMessage[] = []

  helpArticles: SupportArticle[] = []

  contactForm: FormGroup

  constructor(
    private fb: FormBuilder,
    private apiService: ApiService,
  ) {
    this.chatForm = this.fb.group({
      message: ["", Validators.required],
    })

    this.contactForm = this.fb.group({
      name: ["", Validators.required],
      email: ["", [Validators.required, Validators.email]],
      type: ["technical", Validators.required],
      priority: ["medium", Validators.required],
      subject: ["", Validators.required],
      description: ["", Validators.required],
      includeSystemInfo: [true],
    })
  }

  ngOnInit(): void {
    this.loadSupportArticles()
    this.initializeChat()
  }

  toggleWidget(): void {
    this.isExpanded = !this.isExpanded
  }

  initializeChat(): void {
    const welcomeMessage: ChatMessage = {
      id: "1",
      message: "Hi! I'm here to help you with Enterprise RAG Bot. What can I assist you with today?",
      sender: "agent",
      timestamp: new Date().toISOString(),
      type: "text",
    }
    this.chatHistory = [welcomeMessage]
  }

  sendMessage(): void {
    if (this.chatForm.valid) {
      const message = this.chatForm.get("message")?.value

      const userMessage: ChatMessage = {
        id: Date.now().toString(),
        message: message,
        sender: "user",
        timestamp: new Date().toISOString(),
        type: "text",
      }

      this.chatHistory.push(userMessage)
      this.chatLoading = true
      this.chatForm.reset()

      this.apiService.sendChatMessage(message).subscribe({
        next: (response: any) => {
          const agentMessage: ChatMessage = {
            id: Date.now().toString() + "_agent",
            message: response.message || "I understand your question. Let me help you with that.",
            sender: "agent",
            timestamp: response.timestamp || new Date().toISOString(),
            type: "text",
          }
          this.chatHistory.push(agentMessage)
          this.scrollToBottom()
          this.chatLoading = false
        },
        error: (error: any) => {
          console.error("Chat error:", error)
          const errorMessage: ChatMessage = {
            id: Date.now().toString() + "_error",
            message: "Sorry, I'm having trouble responding right now. Please try again or contact support directly.",
            sender: "agent",
            timestamp: new Date().toISOString(),
            type: "text",
          }
          this.chatHistory.push(errorMessage)
          this.scrollToBottom()
          this.chatLoading = false
        },
      })
    }
  }

  scrollToBottom(): void {
    setTimeout(() => {
      if (this.chatMessages) {
        const element = this.chatMessages.nativeElement
        element.scrollTop = element.scrollHeight
      }
    }, 100)
  }

  loadSupportArticles(): void {
    this.apiService.getSupportArticles().subscribe({
      next: (response) => {
        this.helpArticles = response.articles
      },
      error: (error) => {
        console.error("Error loading support articles:", error)
        this.helpArticles = [
          {
            id: "1",
            title: "Getting Started with Enterprise RAG Bot",
            content: "Welcome to Enterprise RAG Bot! Here's how to get started...",
            category: "Getting Started",
            tags: ["setup", "quickstart", "tutorial"],
            helpful_count: 45,
          },
        ]
      },
    })
  }

  submitTicket(): void {
    if (this.contactForm.invalid) return

    this.ticketLoading = true
    const ticketData = {
      ...this.contactForm.value,
      systemInfo: this.contactForm.value.includeSystemInfo ? this.getSystemInfo() : null,
      timestamp: new Date().toISOString(),
      status: "open",
    }

    this.apiService.submitSupportTicket(ticketData).subscribe({
      next: (response) => {
        alert(`Support ticket submitted successfully! Ticket ID: ${response.ticket_id}`)
        this.contactForm.reset()
        this.contactForm.patchValue({
          type: "technical",
          priority: "medium",
          includeSystemInfo: true,
        })
        this.ticketLoading = false
      },
      error: (error) => {
        console.error("Ticket submission error:", error)
        alert("Error submitting ticket. Please try again or contact support directly.")
        this.ticketLoading = false
      },
    })
  }

  getSystemInfo(): any {
    return {
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: new Date().toISOString(),
      screenResolution: `${screen.width}x${screen.height}`,
      browserLanguage: navigator.language,
    }
  }
}
