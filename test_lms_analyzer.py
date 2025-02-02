import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lms_content_analyzer import LMSContentAnalyzer
import tempfile
import os

class TestLMSContentAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample data
        current_date = datetime.now()
        sample_data = {
            'completion_count': [6000, 3000, 2000, 5500, 1000],
            'client_percentage': [6.0, 3.0, 2.0, 5.5, 1.0],
            'number_of_clients': [100, 50, 30, 80, 20],
            'training_category': ['Leadership Development', 'Mandatory and Compliance', 
                                'Information Technology and Systems', 'Clinical', 'Safety'],
            'persona_type': ['leaders', 'employees', 'managers', 'employees', 'employees'],
            'function_specific': ['IT', 'pharmacy', 'clinical', 'IT', 'pharmacy'],
            'region_entity': ['NCAL', 'SCAL', 'CO', 'GA', 'KPWA'],
            'market_function': ['role specific', 'role specific', 'context specific', 
                              'role specific', 'context specific'],
            'delivery_method': ['instructor_led_in_person', 'self_paced_elearning',
                              'microlearning', 'instructor_led_virtual', 'microlearning'],
            'duration_minutes': [45, 25, 15, 30, 10],
            'available_from': [
                current_date - timedelta(days=180),
                current_date - timedelta(days=400),
                current_date - timedelta(days=800),
                current_date - timedelta(days=1200),
                current_date - timedelta(days=100)
            ],
            'training_organization': ['Enterprise Learning Design & Delivery', 'Clinical Education',
                                    'TRO', 'Market L&D teams', 'NEH&S'],
            'content_source': ['in_house', 'custom_vendor', 'url_internal',
                             'off_the_shelf', 'coordinator_deployed'],
            'assignment_type': ['self_assigned', 'assigned', 'self_assigned',
                              'assigned', 'assigned'],
            'training_hours': [2.5, 1.0, 0.5, 1.5, 0.3],
            'interest_area': ['Leadership', 'Compliance', 'Technology',
                            'Clinical Skills', 'Safety'],
            'interest_percentage': [85, 70, 65, 75, 60]
        }
        
        # Create temporary Excel file
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_file = os.path.join(cls.temp_dir, 'test_data.xlsx')
        pd.DataFrame(sample_data).to_excel(cls.test_file, index=False)
        
        # Initialize analyzer
        cls.analyzer = LMSContentAnalyzer(cls.test_file)

    def test_education_history(self):
        history = self.analyzer.analyze_education_history()
        self.assertIn('completion_tracking', history)
        self.assertEqual(history['completion_tracking']['high_volume_count'], 2)
        self.assertEqual(history['completion_tracking']['total_completions'], 17500)

    def test_training_categories(self):
        categories = self.analyzer.analyze_training_categories()
        self.assertTrue(len(categories) > 0)
        self.assertIn('Leadership Development', categories)
        self.assertEqual(categories['Leadership Development']['count'], 1)

    def test_training_focus(self):
        focus = self.analyzer.analyze_training_focus()
        self.assertIn('persona_distribution', focus)
        self.assertIn('function_specific', focus)
        self.assertTrue(focus['persona_distribution']['employees']['percentage'] > 0)

    def test_training_breadth(self):
        breadth = self.analyzer.analyze_training_breadth()
        self.assertIn('market_function', breadth)
        self.assertTrue(any(breadth['market_function'].values()))

    def test_delivery_methods(self):
        delivery = self.analyzer.analyze_delivery_methods()
        self.assertIn('microlearning', delivery)
        self.assertEqual(delivery['duration_analysis']['microlearning'], 2)

    def test_content_usage(self):
        usage = self.analyzer.analyze_content_usage()
        self.assertIn('1_year', usage)
        self.assertTrue(usage['1_year']['count'] > 0)

    def test_training_volume(self):
        volume = self.analyzer.analyze_training_volume()
        self.assertTrue(len(volume) > 0)
        self.assertIn('Enterprise Learning Design & Delivery', volume)

    def test_content_production(self):
        production = self.analyzer.analyze_content_production()
        self.assertTrue(len(production) > 0)
        self.assertIn('in_house', production)

    def test_training_assignment(self):
        assignment = self.analyzer.analyze_training_assignment()
        self.assertIn('self_assigned', assignment)
        self.assertIn('assigned', assignment)
        self.assertTrue(assignment['hours_by_group'])

    def test_learner_interests(self):
        interests = self.analyzer.analyze_learner_interests()
        self.assertTrue(len(interests['top_interests']) > 0)
        self.assertEqual(len(interests['top_interests']), 5)

    def test_report_generation(self):
        report = self.analyzer.generate_enhanced_report()
        self.assertIsInstance(report, str)
        self.assertIn("KP Learn Learning Insights Report", report)
        self.assertIn("Education History Tracking", report)
        self.assertIn("Training Split by Broad Categories", report)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        os.remove(cls.test_file)
        os.rmdir(cls.temp_dir)

if __name__ == '__main__':
    unittest.main(verbosity=2) 