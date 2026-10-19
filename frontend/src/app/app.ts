import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';

export interface BenchmarkRow {
  hotel: string;
  protocol: string;
  model: string;
  auc_roc: number;
  auc_roc_std: number;
  auc_pr: number;
  auc_pr_std: number;
  f1: number;
  f1_std: number;
  accuracy: number;
  accuracy_std: number;
  precision: number;
  precision_std: number;
  recall: number;
  recall_std: number;
  brier: number;
  brier_std: number;
}

export interface TabNetComp {
  hotel: string;
  published_auc_roc: number;
  our_tabnet_auc_roc: number;
  diff_auc_roc: number;
  published_auc_pr: number;
  our_tabnet_auc_pr: number;
  diff_auc_pr: number;
  best_p1_model: string;
  best_p1_auc_roc: number;
}

export interface AblationRow {
  hotel: string;
  features: string;
  auc_roc: number;
  auc_pr: number;
  f1: number;
  accuracy: number;
  precision: number;
  recall: number;
  brier: number;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit {
  // Navigation
  activeTab = signal<'overview' | 'benchmarks' | 'gate1' | 'simulator'>('overview');
  
  // API URL: dynamically resolved from window.__APP_CONFIG__ or Angular environment
  apiUrl: string = '';
  apiOnline = signal<boolean>(false);
  apiStatusText = signal<string>('Checking Flask API connection...');
  modelAccuracy = signal<number>(0.8757);
  modelAucRoc = signal<number>(0.9484);

  // Authentic Research Data Signals
  benchmarks = signal<BenchmarkRow[]>([]);
  tabnetData = signal<TabNetComp[]>([]);
  ablationData = signal<AblationRow[]>([]);
  overviewData = signal<any>(null);
  dataLoaded = signal<boolean>(false);

  // Filters
  selectedHotel = signal<string>('ALL');
  selectedProtocol = signal<string>('ALL');
  selectedMetric = signal<string>('auc_roc');

  // Simulator Form State
  simForm = {
    hotel: 'City Hotel',
    lead_time: 145,
    deposit_type: 'No Deposit',
    customer_type: 'Transient',
    market_segment: 'Online TA',
    distribution_channel: 'TA/TO',
    reserved_room_type: 'A',
    meal: 'BB',
    adr: 110,
    total_of_special_requests: 1,
    required_car_parking_spaces: 0,
    previous_cancellations: 0,
    is_repeated_guest: 0,
    stays_in_week_nights: 3,
    stays_in_weekend_nights: 2,
    adults: 2,
    children: 0,
    arrival_date_month: 'July'
  };

  // Prediction State (No fake fallback outputs permitted)
  isPredicting = signal<boolean>(false);
  predictionResult = signal<any>(null);
  predictionError = signal<string | null>(null);

  constructor(private http: HttpClient) {
    // Resolve API URL
    if (typeof window !== 'undefined' && (window as any).__APP_CONFIG__?.apiUrl) {
      this.apiUrl = (window as any).__APP_CONFIG__.apiUrl;
    } else {
      this.apiUrl = environment.apiUrl || '';
    }
  }

  ngOnInit() {
    this.initDataAndHealth();
  }

  initDataAndHealth() {
    if (this.apiUrl) {
      // Check Flask API health
      this.http.get<any>(`${this.apiUrl}/api/health`).subscribe({
        next: (res) => {
          this.apiOnline.set(true);
          const aucPct = res.model_auc_roc ? (res.model_auc_roc * 100).toFixed(1) : '94.8';
          this.apiStatusText.set(`Flask API Online · HistGradientBoosting (${aucPct}% AUC Demo Model)`);
          if (res.model_accuracy) this.modelAccuracy.set(res.model_accuracy);
          if (res.model_auc_roc) this.modelAucRoc.set(res.model_auc_roc);
          this.loadDataFromApi();
        },
        error: () => {
          this.apiOnline.set(false);
          this.apiStatusText.set('Flask API Offline · Live Simulator Disabled (Static Research Active)');
          this.loadStaticResearchData();
        }
      });
    } else {
      // Production without configured API URL
      this.apiOnline.set(false);
      this.apiStatusText.set('Static Mode · Flask API Not Configured (Live Simulator Requires Backend)');
      this.loadStaticResearchData();
    }
  }

  checkFlaskHealth() {
    this.initDataAndHealth();
  }

  loadDataFromApi() {
    this.http.get<any>(`${this.apiUrl}/api/benchmarks`).subscribe({
      next: (res) => {
        if (res.benchmarks) {
          this.benchmarks.set(res.benchmarks);
        }
      },
      error: () => this.loadStaticResearchData()
    });

    this.http.get<any>(`${this.apiUrl}/api/tabnet-comparison`).subscribe({
      next: (res) => {
        if (res.comparisons) {
          this.tabnetData.set(res.comparisons);
        }
      }
    });

    this.http.get<any>(`${this.apiUrl}/api/ablation`).subscribe({
      next: (res) => {
        if (res.ablation) {
          this.ablationData.set(res.ablation);
        }
      }
    });

    this.http.get<any>(`${this.apiUrl}/api/overview`).subscribe({
      next: (res) => {
        this.overviewData.set(res);
        this.dataLoaded.set(true);
      }
    });
  }

  loadStaticResearchData() {
    // Load authentic research metrics extracted from the notebook artifact bundled in public/data/metrics.json
    this.http.get<any>('data/metrics.json').subscribe({
      next: (data) => {
        if (data.benchmarks) this.benchmarks.set(data.benchmarks);
        if (data.comparison) this.tabnetData.set(data.comparison);
        if (data.ablation) this.ablationData.set(data.ablation);
        if (data.overview) this.overviewData.set(data.overview);
        this.dataLoaded.set(true);
      },
      error: (err) => {
        console.warn('Could not load static research data:', err);
      }
    });
  }

  setTab(tab: 'overview' | 'benchmarks' | 'gate1' | 'simulator') {
    this.activeTab.set(tab);
  }

  setHotel(hotel: string) {
    this.selectedHotel.set(hotel);
  }

  setProtocol(protocol: string) {
    this.selectedProtocol.set(protocol);
  }

  setMetric(metric: string) {
    this.selectedMetric.set(metric);
  }

  getFilteredBenchmarks(): BenchmarkRow[] {
    return this.benchmarks().filter(b => {
      const matchHotel = this.selectedHotel() === 'ALL' || b.hotel.includes(this.selectedHotel());
      const matchProto = this.selectedProtocol() === 'ALL' || b.protocol.includes(this.selectedProtocol());
      return matchHotel && matchProto;
    }).sort((a, b) => {
      const key = this.selectedMetric() as keyof BenchmarkRow;
      return (b[key] as number) - (a[key] as number);
    });
  }

  getMetricValue(row: BenchmarkRow): number {
    const key = this.selectedMetric() as keyof BenchmarkRow;
    return row[key] as number;
  }

  getMetricLabel(metricKey: string): string {
    switch (metricKey) {
      case 'auc_roc': return 'AUC-ROC Score';
      case 'auc_pr': return 'AUC-PR (Avg Precision)';
      case 'f1': return 'F1 Score';
      case 'accuracy': return 'Accuracy Rate';
      case 'precision': return 'Precision';
      case 'recall': return 'Recall Rate';
      case 'brier': return 'Brier Score Loss (Lower is better)';
      default: return metricKey;
    }
  }

  applyPreset(preset: string) {
    if (preset === 'high-risk') {
      this.simForm = {
        hotel: 'City Hotel',
        lead_time: 260,
        deposit_type: 'Non Refund',
        customer_type: 'Transient',
        market_segment: 'Online TA',
        distribution_channel: 'TA/TO',
        reserved_room_type: 'A',
        meal: 'BB',
        adr: 130,
        total_of_special_requests: 0,
        required_car_parking_spaces: 0,
        previous_cancellations: 2,
        is_repeated_guest: 0,
        stays_in_week_nights: 4,
        stays_in_weekend_nights: 2,
        adults: 2,
        children: 0,
        arrival_date_month: 'August'
      };
    } else if (preset === 'safe-resort') {
      this.simForm = {
        hotel: 'Resort Hotel',
        lead_time: 14,
        deposit_type: 'No Deposit',
        customer_type: 'Transient',
        market_segment: 'Direct',
        distribution_channel: 'Direct',
        reserved_room_type: 'C',
        meal: 'HB',
        adr: 180,
        total_of_special_requests: 2,
        required_car_parking_spaces: 1,
        previous_cancellations: 0,
        is_repeated_guest: 1,
        stays_in_week_nights: 5,
        stays_in_weekend_nights: 2,
        adults: 2,
        children: 1,
        arrival_date_month: 'June'
      };
    } else if (preset === 'corporate') {
      this.simForm = {
        hotel: 'City Hotel',
        lead_time: 6,
        deposit_type: 'No Deposit',
        customer_type: 'Transient',
        market_segment: 'Corporate',
        distribution_channel: 'Corporate',
        reserved_room_type: 'A',
        meal: 'BB',
        adr: 95,
        total_of_special_requests: 1,
        required_car_parking_spaces: 1,
        previous_cancellations: 0,
        is_repeated_guest: 1,
        stays_in_week_nights: 2,
        stays_in_weekend_nights: 0,
        adults: 1,
        children: 0,
        arrival_date_month: 'October'
      };
    }

    if (this.apiOnline()) {
      this.runPrediction();
    } else {
      this.predictionResult.set(null);
      this.predictionError.set('Backend API is offline or not configured. Start the Flask backend (python backend/app.py) or deploy the API to enable live inference.');
    }
  }

  runPrediction() {
    if (!this.apiUrl || !this.apiOnline()) {
      this.predictionResult.set(null);
      this.predictionError.set(
        'Flask API is offline or unconfigured. Live machine learning predictions require an active Flask backend service. ' +
        'To enable live inference, start the Flask backend locally (python backend/app.py) or deploy the API to a cloud service and configure the API URL.'
      );
      return;
    }

    this.isPredicting.set(true);
    this.predictionError.set(null);

    this.http.post<any>(`${this.apiUrl}/api/predict`, this.simForm).subscribe({
      next: (res) => {
        if (res.success) {
          this.predictionResult.set(res);
          this.predictionError.set(null);
        } else {
          this.predictionError.set(res.error || 'Prediction calculation failed.');
          this.predictionResult.set(null);
        }
        this.isPredicting.set(false);
      },
      error: (err) => {
        this.isPredicting.set(false);
        this.predictionResult.set(null);
        const msg = err.error?.error || err.message || 'Failed to connect to Flask prediction service.';
        this.predictionError.set(`Prediction Error: ${msg}`);
      }
    });
  }
}
