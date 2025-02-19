// API Configuration
const API_PROTOCOL = 'http'
const API_HOST = 'localhost'
const API_PORT = '5000'
export const API_BASE_URL = `${API_PROTOCOL}://${API_HOST}:${API_PORT}`

// API Endpoints
export const API_ENDPOINTS = {
  PROFIT_HISTORY: `${API_BASE_URL}/api/profit-history`,
  TRADING_DATA: `${API_BASE_URL}/api/trading-data`,
  MARKET_ANALYSIS: `${API_BASE_URL}/api/market-analysis`,
  TRADING_STATUS: `${API_BASE_URL}/api/trading-status`,
  ENABLE_TRADING: `${API_BASE_URL}/api/enable-trading`,
  DISABLE_TRADING: `${API_BASE_URL}/api/disable-trading`,
  TOGGLE_TRADING: `${API_BASE_URL}/api/trading-data/toggle`,
}

// API Fetch Configuration
export const API_CONFIG = {
  DEFAULT_HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
}

// Helper function to handle API errors
export const handleApiError = (error: any) => {
  if (error.name === 'AbortError') {
    return { error: 'Request timed out. Please try again.' }
  }
  if (!navigator.onLine) {
    return { error: 'No internet connection. Please check your network.' }
  }
  if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
    return { error: 'Could not connect to the server. Please ensure the trading bot is running.' }
  }
  if (error.response) {
    return { error: `Server error: ${error.response.status} - ${error.response.statusText}` }
  }
  return { error: 'An unexpected error occurred. Please try again.' }
}

// Helper function for API requests with timeout and retry
export const fetchWithRetry = async <T>(
  url: string, 
  options: RequestInit = {}, 
  retries = API_CONFIG.RETRY_ATTEMPTS
): Promise<T> => {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT)

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...API_CONFIG.DEFAULT_HEADERS,
        ...options.headers,
      },
      signal: controller.signal,
      credentials: 'include', // Include credentials for CORS
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    if (!data) {
      throw new Error('Invalid response format')
    }
    return data as T
  } catch (error) {
    if (retries > 0 && (
      error instanceof TypeError && error.message.includes('Failed to fetch') ||
      error instanceof Error && error.message.includes('Invalid response format')
    )) {
      await new Promise(resolve => setTimeout(resolve, API_CONFIG.RETRY_DELAY))
      return fetchWithRetry<T>(url, options, retries - 1)
    }
    throw error
  } finally {
    clearTimeout(timeout)
  }
} 