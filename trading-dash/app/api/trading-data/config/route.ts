import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const config = await req.json()
    
    // Convert percentage values back to decimals
    const normalizedConfig = {
      trading_config: {
        risk_per_trade: config.riskPercentage / 100,
        max_daily_risk: config.maxDailyRisk / 100,
        symbols: config.enabledPairs,
      },
      session_config: {
        asia_session: {
          enabled: config.sessionRules.asia,
          // Preserve other session settings
          start: "00:00",
          end: "08:00",
          pairs: config.enabledPairs,
          min_range_pips: 4,
          max_range_pips: 115,
          volatility_factor: 1.0
        },
        london_session: {
          enabled: config.sessionRules.london,
          // Preserve other session settings
          start: "08:00",
          end: "16:00",
          pairs: config.enabledPairs,
          min_range_pips: 5,
          max_range_pips: 173,
          volatility_factor: 1.2
        },
        new_york_session: {
          enabled: config.sessionRules.newYork,
          // Preserve other session settings
          start: "13:00",
          end: "21:00",
          pairs: config.enabledPairs,
          min_range_pips: 5,
          max_range_pips: 173,
          volatility_factor: 1.2
        }
      },
      signal_thresholds: {
        strong: config.signalThresholds.strong / 100,
        moderate: config.signalThresholds.moderate / 100,
        weak: config.signalThresholds.weak / 100,
        minimum: config.signalThresholds.minimum / 100
      }
    }
    
    // Send the configuration update to the Python backend
    const pythonResponse = await fetch('http://localhost:8000/api/update-config', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(normalizedConfig),
    })

    if (!pythonResponse.ok) {
      throw new Error('Failed to update Python backend configuration')
    }

    const result = await pythonResponse.json()
    
    return NextResponse.json({
      success: true,
      config: normalizedConfig,
      message: result.message
    })
    
  } catch (error) {
    console.error('Error updating configuration:', error)
    return NextResponse.json(
      { error: 'Failed to update configuration' },
      { status: 500 }
    )
  }
} 