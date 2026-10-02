import unittest
from comparison.multi_field.adapters import valid_object
from comparison.multi_field.dataset import normalize_banking77,parse_banking77_csv,parse_banking77_labels
from comparison.multi_field.metrics import summarize
from comparison.multi_field.models import StructuredCase,StructuredPrediction
SCRIPT=('x=ClassLabel(names='+repr([f'label_{i}' for i in range(77)])+')').encode()
CSV=b'text,category\r\n"hello, bank",a_one\r\nbye,b_two\r\n'
class Banking77Tests(unittest.TestCase):
 def test_crlf_csv_and_script_labels(self):
  self.assertEqual(parse_banking77_labels(SCRIPT),tuple(f'label_{i}' for i in range(77)))
  self.assertEqual(parse_banking77_csv(CSV),[(0,'hello, bank','a_one'),(1,'bye','b_two')])
  with self.assertRaises(ValueError): normalize_banking77(CSV,SCRIPT) # fixture categories deliberately differ
 def test_enum_validation(self):
  schema={'intent':{'type':'enum','description':'x','choices':['a_one','b_two']}}
  self.assertTrue(valid_object({'intent':'a_one'},schema));self.assertFalse(valid_object({'intent':'other'},schema))
 def test_enum_metrics_including_zero_support_choice(self):
  schema={'intent':{'type':'enum','description':'x','choices':['a','b','c']}}
  c1=StructuredCase('1','',schema,{'intent':'a'});c2=StructuredCase('2','',schema,{'intent':'b'})
  p1=StructuredPrediction('x','m','ok',{'intent':'a'},exact_valid=True);p2=StructuredPrediction('x','m','ok',{'intent':'a'},exact_valid=True)
  result=summarize([(c1,p1),(c2,p2)])
  self.assertEqual(result['top1_accuracy'],.5);self.assertEqual(result['macro_f1_present_gold_classes'],1/3)
  self.assertEqual(result['per_class']['c']['support'],0);self.assertEqual(result['per_class']['c']['f1'],0)
if __name__=='__main__':unittest.main()
