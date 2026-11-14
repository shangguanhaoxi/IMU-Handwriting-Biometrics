import unittest
from omc.data import _identify_character

class DiscoveryTest(unittest.TestCase):
    def setUp(self): self.chars={'一','十','三','人'}
    def test_original_names(self):
        self.assertEqual(_identify_character('一',self.chars),'一')
        self.assertEqual(_identify_character('十',self.chars),'十')
    def test_trial_suffixes(self):
        for stem in ['一_trial001','一_02','一-2','一（2）','一(3)','一 第2次']:
            self.assertEqual(_identify_character(stem,self.chars),'一',stem)
    def test_unknown_name(self):
        self.assertIsNone(_identify_character('测试文件',self.chars))
        self.assertIsNone(_identify_character('一二',self.chars))

if __name__=='__main__': unittest.main()
