# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**kalshi-engines** is a collection of quantitative trading engines designed for the Kalshi prediction market. The project aims to identify and capture alpha in specific niche markets through rigorous modeling and simulation.

The immediate focus is on two primary engines:
1.  **Speech Probability Engine ("Book of Truth")**: Predicting spoken words by public figures based on history, context, and news cycles.
2.  **NFL Prediction Engine**: A drive-based simulation engine for game lines, spreads, and player props.

## Architecture

### 1. Speech Probability Engine ("Book of Truth")
Target Market: [Kalshi Mentions](https://kalshi.com/?category=mentions)
A hybrid architecture combining historical patterns with real-time signals.
*   **Historical Speech Frequency Model (HSFM)**: Embedding + RNN/Transformer classifier. Learns speaker-specific base rates (P(word|speaker)).
*   **Contextual Intent Model (CIM)**: Fine-tuned LLM or classifier. Analyzes event context (agendas, crisis status) to predict relevance (P(word|context)).
*   **Real-Time News Momentum Model (RNMM)**: Sentiment/semantic encoder processing recent news (last 14 days) to detect topic "heat" (P(word|news_cycle)).
*   **Ensemble Layer**: Combines probabilities: `P_final = w1*HSFM + w2*CIM + w3*RNMM`.

### 2. NFL Prediction Engine
Target Market: [Kalshi Football](https://kalshi.com/sports/football)
A bottom-up simulation approach rather than top-down regression.
*   **Data Layer**: Ingests play-by-play (nflfastR), drive data, EPA values, weather, injuries, and depth charts.
*   **Feature Layer**:
    *   *Team*: EPA/play (offense/defense), Red zone efficiency, Pressure rates.
    *   *Player*: QB EPA/QBR, Snap shares, Target shares.
    *   *Game*: Weather (wind), Venue (home/away, indoor/outdoor).
*   **Model Layer**:
    *   *Drive Outcome*: Probabilities for TD/FG/Punt/Turnover.
    *   *Game Script*: Pace and pass/run ratios.
    *   *Team Strength*: Weekly updated Elo/EPA hybrid.
    *   *Player Props*: XGBoost/LightGBM models for yardage/TDs.
*   **Simulation Layer**: Monte Carlo simulation (10,000+ runs) of full games drive-by-drive to generate probability distributions for spreads, totals, and props.

## Current Plan

### Phase 0: Foundation
- [ ] Initialize repository structure (Python project).
- [ ] Set up dependency management (poetry or pip).

### Phase 1: Speech Engine Prototype
- [ ] **Data**: Implement scrapers for transcripts and news feeds.
- [ ] **Models**: Build baseline HSFM and RNMM.
- [ ] **Inference**: Create a script to output probabilities for a target word list.

### Phase 2: NFL Engine Prototype
- [ ] **Data**: Set up `nflfastR` data pipeline and EPA calculations.
- [ ] **Sim**: Build the core Drive-Based Monte Carlo simulator.
- [ ] **Props**: Implement basic player prop models.

### Phase 3: Submission Package
- [ ] **Docs**: Write README.md with design decisions, dependencies, and limitations.
- [ ] **Site**: Build bare-bones HTML site linking to repo/docs with a project summary.
- [ ] **Demo**: Create a script/notebook producing sample prediction output.

## Development Commands

*To be populated as tooling is selected (e.g., `poetry run`, `pytest`).*

## Key Files & Structure

*   `speech_engine/`: Components for the Book of Truth.
*   `nfl_engine/`: Components for the NFL simulator.
*   `submission_site/`: Static HTML assets for the submission requirement.
