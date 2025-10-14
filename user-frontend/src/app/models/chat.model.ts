export interface UserQueryRequest {
  query: string;
  max_results?: number;
  include_sources?: boolean;
  search_depth?: 'quick' | 'balanced' | 'deep';
}

export interface ChatResponse {
  query: string;
  answer: string;
  steps?: Array<{ index?: number; text: string; type?: string; image?: any }>;
  summary?: string;
  images?: Array<any>;
  sources?: Array<any>;
  has_sources?: boolean;
  confidence?: number;
  timestamp?: string;
}
