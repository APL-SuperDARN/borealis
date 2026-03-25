import os
import unittest
from multiprocessing import shared_memory

import numpy as np

os.environ.setdefault("BOREALISPATH", "/home/radar/borealis")
os.environ.setdefault("RADAR_ID", "wal")

from rx_signal_processing import _need_intf_beamforming, _need_main_beamforming
from utils.data_aggregator import Aggregator
from utils.message_formats import DebugDataStage, OutputDataset, ProcessedSequenceMessage


class DummyRxParams:
    def __init__(self, *, acf=False, xcf=False, acfint=False, intf_antennas=None):
        self.acf = acf
        self.xcf = xcf
        self.acfint = acfint
        self.intf_antennas = set() if intf_antennas is None else set(intf_antennas)


class TestBeamformingNeeds(unittest.TestCase):
    def test_beamforming_not_required_for_antenna_only_output(self):
        rx_params = DummyRxParams(intf_antennas={0, 1})
        self.assertFalse(_need_main_beamforming(rx_params, enable_bfiq=False))
        self.assertFalse(_need_intf_beamforming(rx_params, enable_bfiq=False))

    def test_beamforming_required_for_requested_products(self):
        self.assertTrue(
            _need_main_beamforming(DummyRxParams(acf=True), enable_bfiq=False)
        )
        self.assertTrue(
            _need_main_beamforming(DummyRxParams(xcf=True), enable_bfiq=False)
        )
        self.assertTrue(
            _need_main_beamforming(DummyRxParams(), enable_bfiq=True)
        )
        self.assertTrue(
            _need_intf_beamforming(
                DummyRxParams(acfint=True, intf_antennas={0}), enable_bfiq=False
            )
        )
        self.assertTrue(
            _need_intf_beamforming(
                DummyRxParams(xcf=True, intf_antennas={0}), enable_bfiq=False
            )
        )
        self.assertTrue(
            _need_intf_beamforming(
                DummyRxParams(intf_antennas={0}), enable_bfiq=True
            )
        )


class TestAggregatorOptionalBfiq(unittest.TestCase):
    def test_aggregator_accepts_antenna_only_sequence(self):
        samples = (
            np.arange(45, dtype=np.float32).reshape(1, 15, 3)
            + 1j * np.zeros((1, 15, 3), dtype=np.float32)
        ).astype(np.complex64)
        shm = shared_memory.SharedMemory(create=True, size=samples.nbytes)
        shm_array = np.ndarray(samples.shape, dtype=np.complex64, buffer=shm.buf)
        shm_array[...] = samples

        processed_data = ProcessedSequenceMessage()
        processed_data.sequence_num = 7
        processed_data.gps_locked = True
        processed_data.gps_to_system_time_diff = 0.0
        processed_data.agc_status_bank_h = 0
        processed_data.lp_status_bank_h = 0
        processed_data.rawrf_shm = ""
        processed_data.debug_data = [
            DebugDataStage(stage_name="antennas", main_shm=shm.name, num_samps=3)
        ]
        processed_data.output_datasets = [OutputDataset(0, 1, 1, 1)]

        aggregator = Aggregator(
            num_main_antennas=15,
            rx_main_antennas=list(range(15)),
            rx_intf_antennas=[],
        )
        aggregator.update(processed_data)

        self.assertTrue(aggregator.antennas_iq_available)
        self.assertFalse(aggregator.bfiq_available)
        self.assertIn(0, aggregator.antenna_iq_accumulator)
        self.assertIn("antennas", aggregator.antenna_iq_accumulator[0])
        first_antenna = aggregator.antenna_iq_accumulator[0]["antennas"][0]
        self.assertEqual(len(first_antenna), 1)
        self.assertEqual(first_antenna[0].shape, (3,))
        shm.close()


if __name__ == "__main__":
    unittest.main()
