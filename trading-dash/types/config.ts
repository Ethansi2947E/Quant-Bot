export interface SessionConfig {
  asia_session: {
    enabled: boolean;
    start: string;
    end: string;
    pairs: string[];
    min_range_pips: number;
    max_range_pips: number;
    volatility_factor: number;
  };
  london_session: {
    enabled: boolean;
    start: string;
    end: string;
    pairs: string[];
    min_range_pips: number;
    max_range_pips: number;
    volatility_factor: number;
  };
  new_york_session: {
    enabled: boolean;
    start: string;
    end: string;
    pairs: string[];
    min_range_pips: number;
    max_range_pips: number;
    volatility_factor: number;
  };
}

export const SESSION_CONFIG: SessionConfig = {
  asia_session: {
    enabled: true,
    start: "00:00",
    end: "08:00",
    pairs: [],  // This will be populated from the API
    min_range_pips: 4,
    max_range_pips: 115,
    volatility_factor: 1.0
  },
  london_session: {
    enabled: true,
    start: "08:00",
    end: "16:00",
    pairs: [],  // This will be populated from the API
    min_range_pips: 5,
    max_range_pips: 173,
    volatility_factor: 1.2
  },
  new_york_session: {
    enabled: true,
    start: "13:00",
    end: "21:00",
    pairs: [],  // This will be populated from the API
    min_range_pips: 5,
    max_range_pips: 173,
    volatility_factor: 1.2
  }
}; 