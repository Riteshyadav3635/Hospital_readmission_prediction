import { useEffect, useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import axios from 'axios';

const schema = z.object({
  patient_reference: z.string().optional(),
  race: z.string().default('Caucasian'),
  gender: z.string().default('Female'),
  age_group: z.string().default('[50-60)'),
  admission_type_id: z.number().int().min(1).default(1),
  discharge_disposition_id: z.number().int().min(1).default(1),
  admission_source_id: z.number().int().min(1).default(7),
  time_in_hospital: z.number().int().min(1).max(14).default(4),
  medical_specialty: z.string().default('InternalMedicine'),
  num_lab_procedures: z.number().int().min(0).default(43),
  num_procedures: z.number().int().min(0).default(1),
  num_medications: z.number().int().min(0).default(16),
  number_outpatient: z.number().int().min(0).default(0),
  number_emergency: z.number().int().min(0).default(0),
  number_inpatient: z.number().int().min(0).default(0),
  diag_1: z.string().default('276'),
  diag_2: z.string().default('133'),
  diag_3: z.string().default('86'),
  number_diagnoses: z.number().int().min(1).default(8),
  metformin: z.string().default('No'),
  repaglinide: z.string().default('No'),
  nateglinide: z.string().default('No'),
  chlorpropamide: z.string().default('No'),
  glimepiride: z.string().default('No'),
  acetohexamide: z.string().default('No'),
  glipizide: z.string().default('No'),
  glyburide: z.string().default('No'),
  tolbutamide: z.string().default('No'),
  pioglitazone: z.string().default('No'),
  rosiglitazone: z.string().default('No'),
  acarbose: z.string().default('No'),
  miglitol: z.string().default('No'),
  troglitazone: z.string().default('No'),
  tolazamide: z.string().default('No'),
  insulin: z.string().default('No'),
  glyburide_metformin: z.string().default('No'),
  glipizide_metformin: z.string().default('No'),
  glimepiride_pioglitazone: z.string().default('No'),
  metformin_rosiglitazone: z.string().default('No'),
  metformin_pioglitazone: z.string().default('No'),
  change: z.string().default('No'),
  diabetesMed: z.string().default('Yes'),
})

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

type FormValues = z.infer<typeof schema>
type PredictionResult = {
  prediction: number
  probability: number
  risk_level: string
  patient_reference: string | null
  model_features: Record<string, number>
  created_at: string | null
}
type PredictionRecord = Omit<PredictionResult, 'model_features'> & { id: number }

function getErrorMessage(error: unknown) {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail || 'Prediction request failed.'
  }
  return 'Prediction request failed.'
}

function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<PredictionRecord[]>([])
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      patient_reference: 'PT-001',
      race: 'Caucasian',
      gender: 'Female',
      age_group: '[50-60)',
      admission_type_id: 1,
      discharge_disposition_id: 1,
      admission_source_id: 7,
      time_in_hospital: 4,
      medical_specialty: 'InternalMedicine',
      num_lab_procedures: 43,
      num_procedures: 1,
      num_medications: 16,
      number_outpatient: 0,
      number_emergency: 0,
      number_inpatient: 0,
      diag_1: '276',
      diag_2: '133',
      diag_3: '86',
      number_diagnoses: 8,
      metformin: 'No',
      repaglinide: 'No',
      nateglinide: 'No',
      chlorpropamide: 'No',
      glimepiride: 'No',
      acetohexamide: 'No',
      glipizide: 'No',
      glyburide: 'No',
      tolbutamide: 'No',
      pioglitazone: 'No',
      rosiglitazone: 'No',
      acarbose: 'No',
      miglitol: 'No',
      troglitazone: 'No',
      tolazamide: 'No',
      insulin: 'No',
      glyburide_metformin: 'No',
      glipizide_metformin: 'No',
      glimepiride_pioglitazone: 'No',
      metformin_rosiglitazone: 'No',
      metformin_pioglitazone: 'No',
      change: 'No',
      diabetesMed: 'Yes',
    },
  })

  async function refreshHistory() {
    try {
      const response = await axios.get<PredictionRecord[]>(`${API_BASE_URL}/api/predictions?limit=10`)
      setHistory(response.data)
      setHistoryError(null)
    } catch {
      setHistoryError('Prediction history could not be loaded.')
    }
  }

  useEffect(() => {
    void refreshHistory()
  }, [])

  async function onSubmit(values: FormValues) {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post<PredictionResult>(`${API_BASE_URL}/api/predict`, values)
      setResult(response.data)
      await refreshHistory()
    } catch (requestError: unknown) {
      setError(getErrorMessage(requestError))
    } finally {
      setLoading(false)
    }
  }

  const { register, handleSubmit, formState: { errors } } = form

  return (
    <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8 rounded-2xl border border-indigo-500/20 bg-slate-900/60 p-8 shadow-lg shadow-indigo-500/10">
          <p className="mb-3 inline-flex rounded-full border border-indigo-400/30 bg-indigo-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-indigo-200">
            Clinical ML · Live API
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-white">Hospital Readmission Prediction</h1>
          <p className="mt-4 max-w-3xl text-slate-300">
            Submit patient information to the existing PCA + logistic regression pipeline running in the backend. The frontend never loads the model locally.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.4fr_0.6fr]">
          <form onSubmit={handleSubmit(onSubmit)} className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/30">
            <h2 className="mb-6 text-xl font-semibold text-white">Patient details</h2>

            <div className="grid gap-5 md:grid-cols-2">
              <label className="text-sm text-slate-300">
                Patient Reference
                <input {...register('patient_reference')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white outline-none ring-0" />
              </label>
              <label className="text-sm text-slate-300">
                Race
                <select {...register('race')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>Caucasian</option>
                  <option>AfricanAmerican</option>
                  <option>Asian</option>
                  <option>Hispanic</option>
                  <option>Other</option>
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Gender
                <select {...register('gender')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>Female</option>
                  <option>Male</option>
                  <option>Unknown/Invalid</option>
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Age group
                <select {...register('age_group')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  {[...Array(10)].map((_, index) => (
                    <option key={index}>{`[${index * 10}-${(index + 1) * 10})`}</option>
                  ))}
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Admission type ID
                <input type="number" {...register('admission_type_id', { valueAsNumber: true })} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>
              <label className="text-sm text-slate-300">
                Discharge disposition ID
                <input type="number" {...register('discharge_disposition_id', { valueAsNumber: true })} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>
              <label className="text-sm text-slate-300">
                Admission source ID
                <input type="number" {...register('admission_source_id', { valueAsNumber: true })} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>
              <label className="text-sm text-slate-300">
                Time in hospital
                <input type="number" {...register('time_in_hospital', { valueAsNumber: true })} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>
              <label className="text-sm text-slate-300 md:col-span-2">
                Medical specialty
                <input {...register('medical_specialty')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>

              {([
                ['num_lab_procedures', 'Lab procedures'],
                ['num_procedures', 'Procedures'],
                ['num_medications', 'Medications'],
                ['number_outpatient', 'Outpatient'],
                ['number_emergency', 'Emergency'],
                ['number_inpatient', 'Inpatient'],
                ['number_diagnoses', 'Diagnoses'],
              ] as const).map(([name, label]) => (
                <label key={name} className="text-sm text-slate-300">
                  {label}
                  <input type="number" {...register(name, { valueAsNumber: true })} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
                </label>
              ))}

              <label className="text-sm text-slate-300 md:col-span-2">
                Primary diagnosis code
                <input {...register('diag_1')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
              </label>

              <label className="text-sm text-slate-300 md:col-span-2">
                Secondary diagnoses
                <div className="mt-1 grid gap-3 md:grid-cols-2">
                  <input {...register('diag_2')} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
                  <input {...register('diag_3')} className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white" />
                </div>
              </label>

              <label className="text-sm text-slate-300">
                Metformin
                <select {...register('metformin')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>No</option>
                  <option>Steady</option>
                  <option>Up</option>
                  <option>Down</option>
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Insulin
                <select {...register('insulin')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>No</option>
                  <option>Steady</option>
                  <option>Up</option>
                  <option>Down</option>
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Change
                <select {...register('change')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>Ch</option>
                  <option>No</option>
                </select>
              </label>
              <label className="text-sm text-slate-300">
                Diabetes medication
                <select {...register('diabetesMed')} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-white">
                  <option>Yes</option>
                  <option>No</option>
                </select>
              </label>
            </div>

            {Object.keys(errors).length > 0 && (
              <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                Please correct the form and try again.
              </div>
            )}

            <button type="submit" disabled={loading} className="mt-6 w-full rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60">
              {loading ? 'Running prediction...' : 'Run prediction'}
            </button>
          </form>

          <aside className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/30">
            <h2 className="mb-4 text-xl font-semibold text-white">Result</h2>
            {error && <div role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</div>}
            {!result && !error && <p className="text-slate-400">No prediction yet.</p>}
            {result && (
              <div className="space-y-4 text-sm text-slate-200">
                <div className={`rounded-xl border p-4 ${result.risk_level === 'High Risk of Readmission' ? 'border-rose-500/40 bg-rose-500/10' : 'border-emerald-500/40 bg-emerald-500/10'}`}>
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-300">Risk level</div>
                  <div className="mt-2 text-2xl font-bold text-white">{result.risk_level}</div>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-950 p-4">
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-400">Probability</div>
                  <div className="mt-2 text-3xl font-bold text-indigo-300">{(result.probability * 100).toFixed(1)}%</div>
                </div>
                <div className="rounded-xl border border-slate-700 bg-slate-950 p-4">
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-400">Prediction</div>
                  <div className="mt-2 text-xl font-semibold text-white">{result.prediction === 1 ? 'Readmission likely' : 'Readmission unlikely'}</div>
                </div>
              </div>
            )}
          </aside>
        </div>

        <section aria-labelledby="history-heading" className="mt-8 rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/30">
          <h2 id="history-heading" className="mb-4 text-xl font-semibold text-white">Recent predictions</h2>
          {historyError && <p role="alert" className="text-sm text-red-200">{historyError}</p>}
          {!historyError && history.length === 0 && <p className="text-slate-400">No saved predictions yet.</p>}
          {history.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[36rem] text-left text-sm">
                <thead className="text-slate-400">
                  <tr>
                    <th className="pb-3 pr-4 font-medium">Reference</th>
                    <th className="pb-3 pr-4 font-medium">Risk</th>
                    <th className="pb-3 pr-4 font-medium">Probability</th>
                    <th className="pb-3 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((record) => (
                    <tr key={record.id} className="border-t border-slate-800 text-slate-200">
                      <td className="py-3 pr-4">{record.patient_reference || '—'}</td>
                      <td className="py-3 pr-4">{record.risk_level}</td>
                      <td className="py-3 pr-4">{(record.probability * 100).toFixed(1)}%</td>
                      <td className="py-3">{new Date(record.created_at || '').toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default App
