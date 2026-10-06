import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axios from 'axios'
import App from './App'

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    isAxiosError: vi.fn(() => false),
  },
}))

const savedPrediction = {
  prediction: 0,
  probability: 0.4156,
  risk_level: 'Low Risk of Readmission',
  patient_reference: 'PT-001',
  created_at: '2026-10-06T10:00:00+00:00',
}

describe('prediction form and history', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('submits a prediction, displays the result, and refreshes saved history', async () => {
    vi.mocked(axios.get)
      .mockResolvedValueOnce({ data: [] } as never)
      .mockResolvedValueOnce({ data: [{ ...savedPrediction, id: 1 }] } as never)
    vi.mocked(axios.post).mockResolvedValue({ data: savedPrediction } as never)

    render(<App />)

    await waitFor(() => expect(axios.get).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole('button', { name: 'Run prediction' }))

    expect(await screen.findByText('Low Risk of Readmission')).toBeTruthy()
    expect(within(screen.getByRole('complementary')).getByText('41.6%')).toBeTruthy()
    expect(await screen.findByText('PT-001')).toBeTruthy()
    expect(axios.post).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/predict$/),
      expect.objectContaining({ age_group: '[50-60)' }),
    )
    expect(axios.get).toHaveBeenLastCalledWith(expect.stringMatching(/\/api\/predictions\?limit=10$/))
  })
})
