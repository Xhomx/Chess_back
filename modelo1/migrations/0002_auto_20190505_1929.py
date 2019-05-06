# Generated by Django 2.2 on 2019-05-06 00:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('modelo1', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Tablero',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('posicion', models.CharField(max_length=255)),
            ],
        ),
        migrations.RemoveField(
            model_name='jugada',
            name='posicion',
        ),
        migrations.AlterField(
            model_name='jugada',
            name='movimiento',
            field=models.CharField(max_length=50),
        ),
    ]
