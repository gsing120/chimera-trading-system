"""
IBKR Data Handlers for Chimera Trading System
Handles real-time data from IBKR Gateway
"""

import time
from typing import Dict, Any
from data.data_interface import Level2Update, TradeUpdate, QuoteUpdate


class IBKRDataHandlers:
    """Handles IBKR data updates for the trading system"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.last_update_time = {}
        
    def handle_level2_update(self, update: Level2Update):
        """Handle Level 2 market data updates from IBKR"""
        try:
            symbol = update.symbol
            
            # Update data orchestrator
            if self.trading_system.data_orchestrator:
                self.trading_system.data_orchestrator.process_level2_update(update)
            
            # Get order book
            order_book = self.trading_system.data_orchestrator.get_order_book(symbol)
            if not order_book:
                return
            
            # Calculate features
            features = self.trading_system.feature_engine.update_features(order_book)
            
            # Detect signals
            signals = self.trading_system.signal_detector.detect_signals(
                symbol, order_book, features
            )
            
            # Process signals with ML if enabled
            if signals and not self.trading_system.config.get('no_ml', False):
                self._process_signals_with_ml(symbol, signals, features)
            
            # Update statistics
            self.trading_system.stats['signals_generated'] += len(signals)
            self.last_update_time[symbol] = time.time()
            
            # Log significant updates (throttled)
            current_time = time.time()
            if symbol not in self.last_update_time or \
               current_time - self.last_update_time.get(f"{symbol}_log", 0) > 5:
                
                best_bid = update.bids[0][0] if update.bids else 0
                best_ask = update.asks[0][0] if update.asks else 0
                spread = best_ask - best_bid if best_bid and best_ask else 0
                
                print(f"📊 {symbol}: Bid={best_bid:.2f}, Ask={best_ask:.2f}, "
                      f"Spread={spread:.4f}, Signals={len(signals)}")
                
                self.last_update_time[f"{symbol}_log"] = current_time
                
        except Exception as e:
            print(f"❌ Error processing Level 2 update for {update.symbol}: {e}")
    
    def handle_trade_update(self, trade: TradeUpdate):
        """Handle trade data updates from IBKR"""
        try:
            symbol = trade.symbol
            
            # Update data orchestrator
            if self.trading_system.data_orchestrator:
                self.trading_system.data_orchestrator.process_trade_update(trade)
            
            # Log significant trades
            if trade.size >= 1000:  # Large trades
                print(f"🔥 Large Trade {symbol}: {trade.size} @ ${trade.price:.2f} ({trade.side})")
                
        except Exception as e:
            print(f"❌ Error processing trade update for {trade.symbol}: {e}")
    
    def handle_quote_update(self, quote: QuoteUpdate):
        """Handle quote data updates from IBKR"""
        try:
            symbol = quote.symbol
            
            # Update data orchestrator
            if self.trading_system.data_orchestrator:
                self.trading_system.data_orchestrator.process_quote_update(quote)
                
        except Exception as e:
            print(f"❌ Error processing quote update for {quote.symbol}: {e}")
    
    def _process_signals_with_ml(self, symbol: str, signals: list, features: Dict[str, Any]):
        """Process signals with ML enhancement"""
        try:
            # Market regime detection
            if self.trading_system.regime_detector:
                regime = self.trading_system.regime_detector.predict_regime(features)
                
                # Adjust signal confidence based on regime
                for signal in signals:
                    if hasattr(signal, 'confidence'):
                        if regime in ['trending_up', 'trending_down']:
                            signal.confidence *= 1.2  # Boost trend-following signals
                        elif regime == 'range_bound':
                            signal.confidence *= 0.8  # Reduce trend signals in ranging market
            
            # Signal classification
            if self.trading_system.signal_classifier:
                for signal in signals:
                    ml_confidence = self.trading_system.signal_classifier.classify_signal(
                        signal, features
                    )
                    if hasattr(signal, 'ml_confidence'):
                        signal.ml_confidence = ml_confidence
            
            # RL exit optimization
            if symbol in self.trading_system.rl_agents:
                rl_agent = self.trading_system.rl_agents[symbol]
                for signal in signals:
                    if hasattr(signal, 'entry_price'):
                        # Get RL-optimized exit recommendation
                        exit_action = rl_agent.get_exit_action(
                            signal.entry_price, features
                        )
                        if hasattr(signal, 'rl_exit_action'):
                            signal.rl_exit_action = exit_action
            
            # Update ML statistics
            self.trading_system.stats['ml_predictions'] += len(signals)
            
        except Exception as e:
            print(f"❌ Error in ML processing for {symbol}: {e}")

