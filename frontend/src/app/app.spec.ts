import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { App } from './app';

describe('App', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting()
      ]
    }).compileComponents();
  });

  it('should create the app component', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it('should render the brand title', async () => {
    const fixture = TestBed.createComponent(App);
    await fixture.whenStable();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('h1')?.textContent).toContain('Hotel Cancellation Intelligence');
  });

  it('should switch tabs properly', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    expect(app.activeTab()).toBe('overview');

    app.setTab('benchmarks');
    expect(app.activeTab()).toBe('benchmarks');

    app.setTab('gate1');
    expect(app.activeTab()).toBe('gate1');

    app.setTab('simulator');
    expect(app.activeTab()).toBe('simulator');
  });

  it('should filter benchmarks by hotel and protocol', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;

    app.benchmarks.set([
      { hotel: 'H1 (Resort)', protocol: 'P1 random, raw', model: 'XGBoost', auc_roc: 0.9626, auc_roc_std: 0, auc_pr: 0.91, auc_pr_std: 0, f1: 0.82, f1_std: 0, accuracy: 0.90, accuracy_std: 0, precision: 0.84, precision_std: 0, recall: 0.81, recall_std: 0, brier: 0.06, brier_std: 0 },
      { hotel: 'H2 (City)', protocol: 'P3 time-based, dedup', model: 'CatBoost', auc_roc: 0.8372, auc_roc_std: 0, auc_pr: 0.71, auc_pr_std: 0, f1: 0.61, f1_std: 0, accuracy: 0.76, accuracy_std: 0, precision: 0.69, precision_std: 0, recall: 0.55, recall_std: 0, brier: 0.16, brier_std: 0 }
    ]);

    app.setHotel('H1');
    expect(app.getFilteredBenchmarks().length).toBe(1);
    expect(app.getFilteredBenchmarks()[0].hotel).toContain('H1');

    app.setHotel('ALL');
    app.setProtocol('P3');
    expect(app.getFilteredBenchmarks().length).toBe(1);
    expect(app.getFilteredBenchmarks()[0].protocol).toContain('P3');
  });

  it('should update form values when applying presets', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;

    app.applyPreset('high-risk');
    expect(app.simForm.deposit_type).toBe('Non Refund');
    expect(app.simForm.lead_time).toBe(260);

    app.applyPreset('safe-resort');
    expect(app.simForm.deposit_type).toBe('No Deposit');
    expect(app.simForm.hotel).toBe('Resort Hotel');
    expect(app.simForm.required_car_parking_spaces).toBe(1);
  });

  it('should set an honest error message when predicting while API is offline', () => {
    const fixture = TestBed.createComponent(App);
    const app = fixture.componentInstance;
    app.apiOnline.set(false);

    app.runPrediction();

    expect(app.predictionResult()).toBeNull();
    expect(app.predictionError()).toContain('Flask API is offline or unconfigured');
  });
});
