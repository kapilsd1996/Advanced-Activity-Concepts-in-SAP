import { Component, OnInit } from '@angular/core';
import { DataService } from './data.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})

export class AppComponent implements OnInit {
  availableTables: string[] = [];
  selectedTables: string = '';

  constructor(private dataService: DataService) {}

  ngOnInit(): void {
    this.dataService.getTables().subscribe(tables => {
      this.availableTables = tables;
    });
  }

  processTables(): void {
    this.dataService.processTables(this.selectedTables.split(',')).subscribe(response => {
      alert(response.message);
    });
  }

  generateOCEL(): void {
    this.dataService.generateOCEL(this.selectedTables.split(',')).subscribe(response => {
        console.log(this.selectedTables)
        alert(response.message)
  
    });
  }
}