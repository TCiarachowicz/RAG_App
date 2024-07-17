from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField


class QueryForm(FlaskForm):
    query = StringField('Query')
    submit = SubmitField('Ask')

# depreciated