import unittest
import os
import json
from app import create_app, db

class CompaniesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app(config_name="testing")
        self.client = self.app.test_client
        self.company = {'name': 'Cysect'}

        with self.app.app_context():
            db.create_all()

    def test_companies_creation(self):
        res = self.client().post('/companies/', data=self.company)
        self.assertEqual(res.status_code,201)
        self.assertIn('Cysect', str(res.data))

    # def test_api_can_get_all_companies(self):
    #     """Test API can get a company (GET request)."""
    #     res = self.client().post('/companies/', data=self.company)
    #     self.assertEqual(res.status_code, 201)
    #     res = self.client().get('/companies/')
    #     self.assertEqual(res.status_code, 200)
    #     self.assertIn('Cysect', str(res.data))

    def test_api_can_get_company_by_id(self):
        """Test API can get a single company by using it's id."""
        rv = self.client().post('/companies/', data=self.company)
        self.assertEqual(rv.status_code, 201)
        result_in_json = json.loads(rv.data.decode('utf-8').replace("'", "\""))
        result = self.client().get(
            '/companies/{}'.format(result_in_json['id']))
        self.assertEqual(result.status_code, 200)
        self.assertIn('Cysect', str(result.data))

    def test_company_can_be_edited(self):
        """Test API can edit an existing company. (PUT request)"""
        rv = self.client().post(
            '/companies/',
            data={'name': 'Cysect'})
        self.assertEqual(rv.status_code, 201)
        rv = self.client().put(
            '/companies/1',
            data={
                "name": "neostream)"
            })
        self.assertEqual(rv.status_code, 200)
        results = self.client().get('/companies/1')
        self.assertIn('neostream', str(results.data))

    def test_company_deletion(self):
        """Test API can delete an existing company. (DELETE request)."""
        rv = self.client().post(
            '/companies/',
            data={'name': 'Cysect'})
        self.assertEqual(rv.status_code, 201)
        res = self.client().delete('/companies/1')
        self.assertEqual(res.status_code, 200)
        # Test to see if it exists, should return a 404
        result = self.client().get('/companies/1')
        self.assertEqual(result.status_code, 404)

    def tearDown(self):
        """teardown all initialized variables."""
        with self.app.app_context():
            # drop all tables
            db.session.remove()
            db.drop_all()

# Make the tests conveniently executable
if __name__ == "__main__":
    unittest.main()