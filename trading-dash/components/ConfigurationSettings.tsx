"use client"

import { useState, useEffect, useCallback } from "react"
import { Card } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Settings, Percent, Clock, Signal } from "lucide-react"
import { SESSION_CONFIG } from "@/types/config"

interface ConfigType {
  riskPercentage: number;
  maxDailyRisk: number;
  enabledPairs: string[];
  availablePairs: string[];
  sessionRules: {
    asia: boolean;
    london: boolean;
    newYork: boolean;
  };
  signalThresholds: {
    strong: number;
    moderate: number;
    weak: number;
    minimum: number;
  };
}

interface ConfigurationSettingsProps {
  onSave: (config: ConfigType) => void;
}

const ConfigurationSettings: React.FC<ConfigurationSettingsProps> = ({ onSave }) => {
  const [config, setConfig] = useState<ConfigType>({
    riskPercentage: 0,
    maxDailyRisk: 0,
    enabledPairs: [],
    availablePairs: [],
    sessionRules: {
      asia: true,
      london: true,
      newYork: true
    },
    signalThresholds: {
      strong: 0,
      moderate: 0,
      weak: 0,
      minimum: 0
    }
  })
  const [isDirty, setIsDirty] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isInitialized, setIsInitialized] = useState(false)

  // Load initial configuration from the API
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/config')
        if (!response.ok) {
          throw new Error('Failed to fetch configuration')
        }
        
        const data = await response.json()
        
        // Update config with values from the API
        setConfig({
          riskPercentage: data.trading_config.risk_per_trade * 100,
          maxDailyRisk: data.trading_config.max_daily_risk * 100,
          enabledPairs: data.trading_config.symbols,
          availablePairs: data.trading_config.symbols,
          sessionRules: {
            asia: data.session_config.asia_session.enabled,
            london: data.session_config.london_session.enabled,
            newYork: data.session_config.new_york_session.enabled
          },
          signalThresholds: {
            strong: data.signal_thresholds.strong * 100,
            moderate: data.signal_thresholds.moderate * 100,
            weak: data.signal_thresholds.weak * 100,
            minimum: data.signal_thresholds.minimum * 100
          }
        })
        setIsInitialized(true)
      } catch (error) {
        console.error('Error fetching configuration:', error)
      }
    }

    fetchConfig()
  }, [])

  const handleChange = useCallback((key: keyof ConfigType, value: any) => {
    setConfig(prev => ({
      ...prev,
      [key]: value
    }))
    setIsDirty(true)
  }, [])

  const handleSave = useCallback(async () => {
    setIsLoading(true)
    try {
      // Convert the config to the format expected by the API
      const apiConfig = {
        trading_config: {
          risk_per_trade: config.riskPercentage / 100,
          max_daily_risk: config.maxDailyRisk / 100,
          symbols: config.enabledPairs
        },
        session_config: {
          asia_session: {
            ...SESSION_CONFIG.asia_session,
            enabled: config.sessionRules.asia
          },
          london_session: {
            ...SESSION_CONFIG.london_session,
            enabled: config.sessionRules.london
          },
          new_york_session: {
            ...SESSION_CONFIG.new_york_session,
            enabled: config.sessionRules.newYork
          }
        },
        signal_thresholds: {
          strong: config.signalThresholds.strong / 100,
          moderate: config.signalThresholds.moderate / 100,
          weak: config.signalThresholds.weak / 100,
          minimum: config.signalThresholds.minimum / 100
        }
      }

      const response = await fetch('http://localhost:8000/api/update-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(apiConfig)
      })

      if (!response.ok) {
        throw new Error('Failed to update configuration')
      }

      onSave(config)
      setIsDirty(false)
    } catch (error) {
      console.error('Error saving configuration:', error)
    } finally {
      setIsLoading(false)
    }
  }, [config, onSave])

  const togglePair = useCallback((pair: string) => {
    setConfig(prev => ({
      ...prev,
      enabledPairs: prev.enabledPairs.includes(pair)
        ? prev.enabledPairs.filter(p => p !== pair)
        : [...prev.enabledPairs, pair].sort()
    }))
    setIsDirty(true)
  }, [])

  return (
    <div className="space-y-6">
      {/* Risk Management */}
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <div className="flex items-center gap-4 mb-6">
          <div className="h-12 w-12 rounded-xl bg-blue-500/10 flex items-center justify-center">
            <Percent className="h-6 w-6 text-blue-500" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Risk Management</h3>
            <p className="text-sm text-white/60">Configure risk parameters</p>
          </div>
        </div>

        <div className="space-y-6">
          <div>
            <label className="text-sm font-medium text-white/80 mb-2 block">
              Risk Per Trade: {config.riskPercentage}%
            </label>
            <Slider
              value={[config.riskPercentage]}
              onValueChange={([value]) => handleChange('riskPercentage', value)}
              max={5}
              step={0.1}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-sm font-medium text-white/80 mb-2 block">
              Max Daily Risk: {config.maxDailyRisk}%
            </label>
            <Slider
              value={[config.maxDailyRisk]}
              onValueChange={([value]) => handleChange('maxDailyRisk', value)}
              max={15}
              step={0.5}
              className="w-full"
            />
          </div>
        </div>
      </Card>

      {/* Trading Pairs */}
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <div className="flex items-center gap-4 mb-6">
          <div className="h-12 w-12 rounded-xl bg-purple-500/10 flex items-center justify-center">
            <Settings className="h-6 w-6 text-purple-500" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Trading Pairs</h3>
            <p className="text-sm text-white/60">Select active currency pairs</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {config.availablePairs.map((pair) => (
            <Badge
              key={pair}
              variant={config.enabledPairs.includes(pair) ? "default" : "outline"}
              className="cursor-pointer transition-all duration-200 hover:scale-105"
              onClick={() => togglePair(pair)}
            >
              {pair}
            </Badge>
          ))}
        </div>
      </Card>

      {/* Session Rules */}
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <div className="flex items-center gap-4 mb-6">
          <div className="h-12 w-12 rounded-xl bg-green-500/10 flex items-center justify-center">
            <Clock className="h-6 w-6 text-green-500" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Session Rules</h3>
            <p className="text-sm text-white/60">Configure trading sessions</p>
          </div>
        </div>

        <div className="space-y-4">
          {Object.entries(config.sessionRules).map(([session, enabled]) => (
            <div key={session} className="flex items-center justify-between">
              <label className="text-sm font-medium text-white/80 capitalize">
                {session} Session
              </label>
              <Switch
                checked={enabled}
                onCheckedChange={(checked) => {
                  handleChange('sessionRules', {
                    ...config.sessionRules,
                    [session]: checked
                  })
                }}
              />
            </div>
          ))}
        </div>
      </Card>

      {/* Signal Thresholds */}
      <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
        <div className="flex items-center gap-4 mb-6">
          <div className="h-12 w-12 rounded-xl bg-yellow-500/10 flex items-center justify-center">
            <Signal className="h-6 w-6 text-yellow-500" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Signal Thresholds</h3>
            <p className="text-sm text-white/60">Set confidence thresholds</p>
          </div>
        </div>

        <div className="space-y-6">
          {Object.entries(config.signalThresholds).map(([level, value]) => (
            <div key={level}>
              <label className="text-sm font-medium text-white/80 mb-2 block capitalize">
                {level}: {value}%
              </label>
              <Slider
                value={[value]}
                onValueChange={([newValue]) => {
                  handleChange('signalThresholds', {
                    ...config.signalThresholds,
                    [level]: newValue
                  })
                }}
                max={100}
                step={1}
                className="w-full"
              />
            </div>
          ))}
        </div>
      </Card>

      {/* Save Button */}
      {isDirty && (
        <div className="sticky bottom-6 flex justify-end">
          <button
            onClick={handleSave}
            disabled={isLoading}
            className={`px-6 py-3 bg-blue-500 text-white rounded-xl transition-all duration-200 shadow-lg
              ${isLoading 
                ? 'opacity-50 cursor-not-allowed'
                : 'hover:bg-blue-600 hover:shadow-xl hover:-translate-y-0.5'
              }`}
          >
            {isLoading ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      )}
    </div>
  )
}

export default ConfigurationSettings 