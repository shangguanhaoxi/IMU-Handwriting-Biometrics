import tempfile
import unittest
from pathlib import Path

from omc.data import _identify_character, discover_records


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.chars = {'一', '十', '三', '人'}

    def test_original_names(self):
        self.assertEqual(_identify_character('一', self.chars), '一')
        self.assertEqual(_identify_character('十', self.chars), '十')

    def test_trial_suffixes(self):
        for stem in ['一_trial001', '一_02', '一-2', '一（2）', '一(3)', '一 第2次']:
            self.assertEqual(_identify_character(stem, self.chars), '一', stem)

    def test_discover_extracted_directory(self):
        cfg = {
            'train_characters': ['一'],
            'val_characters': ['三'],
            'test_characters': ['十'],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'excel'
            subject_dir = root / 'A'
            subject_dir.mkdir(parents=True)
            for char in ('一', '三', '十'):
                (subject_dir / f'{char}.xlsx').write_bytes(b'placeholder')
            (root / '新表.xlsx').write_bytes(b'metadata')
            records, ignored, subjects = discover_records(root, cfg)
        self.assertEqual(len(records), 3)
        self.assertEqual(subjects, ['A'])
        self.assertEqual({r['split'] for r in records}, {'train', 'val', 'test'})
        self.assertEqual(len(ignored), 1)

    def test_reject_zip_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_zip = Path(tmp) / 'excel.zip'
            fake_zip.write_bytes(b'not a directory')
            with self.assertRaises(NotADirectoryError):
                discover_records(fake_zip, {
                    'train_characters': ['一'],
                    'val_characters': ['三'],
                    'test_characters': ['十'],
                })

    def test_retired_identity_folder_is_rejected(self):
        cfg = {
            'train_characters': ['一'],
            'val_characters': ['三'],
            'test_characters': ['十'],
            'active_subjects': ['A', 'F'],
            'retired_duplicate_subjects': {'B': 'A'},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in ('A', 'F', 'B'):
                folder = root / subject
                folder.mkdir()
                for char in ('一', '三', '十'):
                    (folder / f'{char}.xlsx').write_bytes(b'placeholder')
            with self.assertRaisesRegex(ValueError, 'active_subjects'):
                discover_records(root, cfg)

    def test_only_unique_active_subjects(self):
        cfg = {
            'train_characters': ['一'],
            'val_characters': ['三'],
            'test_characters': ['十'],
            'active_subjects': ['A', 'F'],
            'retired_duplicate_subjects': {'B': 'A'},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject in ('A', 'F'):
                folder = root / subject
                folder.mkdir()
                for char in ('一', '三', '十'):
                    (folder / f'{char}.xlsx').write_bytes(b'placeholder')
            records, ignored, subjects = discover_records(root, cfg)
            self.assertEqual(subjects, ['A', 'F'])
            self.assertEqual(len(records), 6)
            self.assertEqual(ignored, [])

    def test_unknown_name(self):
        self.assertIsNone(_identify_character('测试文件', self.chars))
        self.assertIsNone(_identify_character('一二', self.chars))


if __name__ == '__main__':
    unittest.main()
